import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from fairseq.models import BaseFairseqModel, register_model, register_model_architecture
from fairseq.tasks import FairseqTask, register_task
from dataclasses import dataclass, field
from fairseq.dataclass import FairseqDataclass
from fairseq.criterions import FairseqCriterion, register_criterion
from typing import Optional, List, Any


@dataclass
class UnsupervisedSpeechConfig(FairseqDataclass):
    """
    Configuration for the Unsupervised Speech Task.
    Defines paths for audio data, unlabelled text data, labels type, and language model.
    """
    data: str = field(default="???", metadata={"help": "path to data directory"})
    text_data: str = field(default="???", metadata={"help": "path to unlabelled text"})
    labels: str = field(default="phn", metadata={"help": "label type"})
    kenlm_path: str = field(default="", metadata={"help": "path to kenlm model"})
    sort_by_length: bool = field(default=True, metadata={"help": "sort dataset by length"})
    shuffle: bool = field(default=True, metadata={"help": "shuffle dataset"})
    max_length: Optional[int] = field(default=None, metadata={"help": "max length"})
    unfiltered: bool = field(default=False, metadata={"help": "unfiltered"})
    uppercase: bool = field(default=False, metadata={"help": "uppercase"})
    skipwords: str = field(default="", metadata={"help": "skipwords"})

@dataclass
class Wav2vec_UConfig(FairseqDataclass):
    lexicon: Optional[str] = None
    kenlm_path: Optional[str] = None
    no_softmax: bool = False
    segmentation: Any = None

    """
    Configuration for the Wav2vec-U model architecture.
    Handles GAN training penalties and generator settings.
    """
    input_dim: int = field(default=512, metadata={"help": "input dimension"})
    batch_size: int = field(default=4, metadata={"help": "batch size"}) 
    smoothness_weight: float = field(default=0.5, metadata={"help": "smoothness weight"})
    gradient_penalty: float = field(default=1.5, metadata={"help": "gradient penalty"})
    code_penalty: float = field(default=4.0, metadata={"help": "code penalty"})
    temp: str = field(default="2.0,0.5,0.99", metadata={"help": "temperature schedule as comma-separated string"})

    def __post_init__(self):
        pass

# --- TASK REGISTRATION ---
@register_task('unsupervised_speech', dataclass=UnsupervisedSpeechConfig)
class UnsupervisedSpeechTask(FairseqTask):
    """
    Fairseq task handling dataset preparation and dictionary management.
    Ensures language dictionary mapping and manifest configurations are met.
    """
    def __init__(self, cfg, dictionary):
        super().__init__(cfg)
        self.dictionary = dictionary

    @property
    def target_dictionary(self):
        return self.dictionary

    @property
    def source_dictionary(self):
        return None

    @classmethod
    def setup_task(cls, cfg, **kwargs):
        """
        Loads the dictionary file and initializes the task.
        Searches for 'dict.phn.txt' or 'dict.txt' in data and text directories.
        """
        text_data_dir = os.path.dirname(cfg.text_data)
        
        # Search order: prefer dict.phn.txt over dict.txt
        candidates = [
            os.path.join(cfg.data, "dict.phn.txt"),
            os.path.join(cfg.data, "dict.txt"),
            os.path.join(text_data_dir, "dict.phn.txt"),
            os.path.join(text_data_dir, "dict.txt"),
        ]
        
        dict_path = None
        for candidate in candidates:
            if os.path.exists(candidate):
                dict_path = candidate
                break
        
        if dict_path is None:
            raise FileNotFoundError(
                f"Dictionary not found in searched paths:\n"
                + "\n".join(f"  - {c}" for c in candidates)
            )
        
        print(f"| [Custom Task] Loading dictionary from: {dict_path}")
        dictionary = cls.load_dictionary(dict_path)
        return cls(cfg, dictionary)
    
    def load_dataset(self, split, combine=False, **kwargs):
        """
        Loads the raw audio manifest and transforms it into a FileAudioDataset.
        Expects raw audio since data returned has shape [batch, time].
        """
        from fairseq.data.audio.raw_audio_dataset import FileAudioDataset
        import os

        # This points to /root/wav2vec_unsupervised/data_preparation/english_audio/train.tsv
        manifest_path = os.path.join(self.cfg.data, f"{split}.tsv")
        
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f"Manifest not found at {manifest_path}. Run the 'find' command to create it!")

        # Initiates an iterable raw unlabelled audio mapping
        self.datasets[split] = FileAudioDataset(
            manifest_path=manifest_path,
            sample_rate=16000,
            max_sample_size=250000,
            min_sample_size=32000,
            pad=True,
        )
        
        print(f"| [Custom Task] Successfully loaded {split} manifest.")

# --- DATASET & REAL DATA ---
class UnsupervisedTextDataset(Dataset):
    """
    Reads unlabelled target text data to serve as the ground truth distribution for the GAN discriminator.
    """
    def __init__(self, path, dictionary):
        self.samples = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                # Tokenizes string phonemes into numerical identifiers.
                tokens = dictionary.encode_line(line.strip(), add_if_not_exist=False).long()
                if len(tokens) > 0:
                    self.samples.append(tokens)

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx): return self.samples[idx]

class RealData:
    """
    Wrapper around UnsupervisedTextDataset providing continuous infinite batches of real phoneme text.
    Returns tensors in one-hot encoded representation.
    """
    def __init__(self, cfg, task):
        self.data_path = task.cfg.text_data
        self.batch_size = cfg.batch_size
        self.dictionary = getattr(task, 'dictionary', getattr(task, 'target_dictionary', None))
        self.vocab_size = len(self.dictionary) # Vocabulary size for one-hot dimensions        
        # Only build the text dataset when a valid plain-text file is provided.
        # During inference the text_data may be a directory (binarized) or empty string,
        # in which case we skip loading to allow checkpoint restoration.
        self.loader = None
        self.iterator = None
        if self.data_path and os.path.isfile(self.data_path):
            dataset = UnsupervisedTextDataset(self.data_path, self.dictionary)
            self.loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, 
                                     collate_fn=self._collate_fn, drop_last=True)
            self.iterator = iter(self.loader)

    def _collate_fn(self, batch):
        """
        Applies batch-level padding to match the longest sequence in the batch.
        """
        max_len = max(len(x) for x in batch)
        padded = torch.stack([F.pad(x, (0, max_len - len(x)), value=self.dictionary.pad()) for x in batch])
        return padded

    def _infinite_generator(self):
        while True:
            for batch in self.loader:
                yield F.one_hot(batch, num_classes=self.vocab_size).float()

    def get_batch(self):
        """
        Continuously pulls target data from the infinite yield generator.
        """
        if not hasattr(self, 'batch_generator') or self.batch_generator is None:
            self.batch_generator = self._infinite_generator()
        return next(self.batch_generator)

# --- GENERATOR & DISCRIMINATOR ---
class Generator(nn.Module):
    """
    Maps unsupervised audio inputs to phoneme distributions.
    """
    def __init__(self, input_dim, output_vocab_size):
        super().__init__()
        # Feature extractor for raw 1D audio shapes: [batch_size, sequence_time].
        # Applies downsampling using convolutional strides to build feature vectors.
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(1, input_dim, kernel_size=400, stride=160, padding=200),
            nn.ReLU(),
            nn.Conv1d(input_dim, input_dim, kernel_size=5, stride=2, padding=2)
        )
        # Projection to probability distribution spanning the vocabulary.
        self.network = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, output_vocab_size)
        )

    def forward(self, audio_features, temperature=1.0):
        # Checks if tensor lacks embedding depth dimensions (it implies it is a raw waveform)
        if audio_features.dim() == 2:
            x = audio_features.unsqueeze(1) # Extrapolate channel dim
            x = self.feature_extractor(x)
            x = x.transpose(1, 2) # Format to [batch, frames, embedding_dim]
        else:
            x = audio_features # Expecting PCA features if provided
        logits = self.network(x)
        
        # Uses Gumbel Softmax for discrete one-hot approximations with differentiable gradients.
        return F.gumbel_softmax(logits, tau=temperature, hard=True, dim=-1)

class Discriminator(nn.Module):
    """
    Classifies generator translations vs real phoneme strings to score sequence authenticity.
    """
    def __init__(self, input_vocab_size):
        super().__init__()
        self.network = nn.Sequential(
            # Convolutional layer for sequence analysis
            nn.Conv1d(input_vocab_size, 256, kernel_size=5, padding=2),
            nn.ReLU(),
            # Averages the sequence into a single feature representation
            nn.AdaptiveAvgPool1d(1), 
            nn.Flatten(), # Maps tensor safely back into Linear inputs shapes
            # Outputs score predicting authenticity; no sigmoid used for WGAN compatibility.
            nn.Linear(256, 1)
        )

    def forward(self, x):
        # We enforce structural constraints to pass data sequence to [batch, features(vocab), time(sequence)]
        if x.shape[-1] == self.network[0].in_channels:
            x = x.transpose(1, 2)
        return self.network(x) # Output scalar scoring authenticity.

# --- MODEL WRAPPER ---
@register_model('wav2vec_u', dataclass=Wav2vec_UConfig)
class Wav2vec_U(BaseFairseqModel):
    """
    Fairseq model wrapper consolidating generator, discriminator, and real data under Hydra management.
    """
    @classmethod
    def build_model(cls, cfg, task):
        if hasattr(task, 'dictionary'):
            vocab_size = len(task.dictionary)
        elif hasattr(task, 'target_dictionary'):
            vocab_size = len(task.target_dictionary)
        else:
            raise ValueError("Task does not expose a dictionary or target_dictionary")
        return cls(cfg, vocab_size, getattr(cfg, 'input_dim', 512), task)

    def __init__(self, cfg, vocab_size, input_dim, task):
        super().__init__()
        self.cfg = cfg
        
        # Convert the string back to a list if it's currently a string
        if isinstance(cfg.temp, str):
            self.temp_schedule = [float(x) for x in cfg.temp.split(",")]
        else:
            self.temp_schedule = cfg.temp

        self.real_data = RealData(cfg, task)
        self.generator = Generator(input_dim, vocab_size)
        self.discriminator = Discriminator(vocab_size)

    def get_groups(self):
        """
        Maps parameters to optimizer groups for separate generator and discriminator updates.
        """
        return {
            "generator": self.generator.parameters(),
            "discriminator": self.discriminator.parameters(),
        }

    def forward(self, source, **kwargs):
        # Default forward pass maps audio to predicted phoneme distributions.
        return self.generator(source, temperature=kwargs.get('temperature', 1.0))

@register_model_architecture('wav2vec_u', 'wav2vec_u_base')
def wav2vec_u_base(cfg):
    pass

@register_criterion('wav2vec_u_criterion')
class ModelCriterion(FairseqCriterion):
    """
    Manages GAN adversarial loss calculations for generator and discriminator updates.
    """
    def forward(self, model, sample, reduce=True):
        source = sample['net_input']['source']
        device = source.device
        
        # Generator creates its predicted translation of the audio
        fake_data = model(source)
        # Pulls target phoneme examples from the dataloader
        real_data = model.real_data.get_batch().to(device)

        # Calculates Discriminator performance evaluating fake data explicitly.
        # .detach() prevents gradient flow back into the Generator weights inside this block logic. 
        # This keeps the Generator shielded from negative updates during the Discriminator's feedback phase.
        d_fake_detached = model.discriminator(fake_data.detach())
        
        # Determines genuine text probability scores
        d_real = model.discriminator(real_data)
        
        # WGAN Loss: maximize real scores, minimize fake scores.
        d_loss = d_fake_detached.mean() - d_real.mean()
        
        # Tracks true linked gradient Fake data computations evaluating Generator performance
        d_fake_g = model.discriminator(fake_data)
        # Generator aims to convince Discriminator fake text was real, MAXIMIZING specific output.
        g_loss = -d_fake_g.mean()

        # Composite mechanism for simultaneous gradient backpropagation.
        loss = d_loss + g_loss
        
        sample_size = source.size(0)
        
        # Pushes metrics to standard logging outputs.
        logging_output = {
            "loss": loss.data,
            "d_loss": d_loss.data,
            "g_loss": g_loss.data,
            "ntokens": sample_size,
            "nsentences": sample_size,
            "sample_size": sample_size,
        }
        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate signals from different batches for logging metrics output reporting."""
        from fairseq import metrics
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        metrics.log_scalar("loss", loss_sum)
        metrics.log_scalar("d_loss", sum(log.get("d_loss", 0) for log in logging_outputs))
        metrics.log_scalar("g_loss", sum(log.get("g_loss", 0) for log in logging_outputs))