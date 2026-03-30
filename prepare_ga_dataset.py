import os
import csv
import random
import subprocess
from pathlib import Path
import string
from tqdm import tqdm  # progress bar

# Define paths relative to the Docker container
BASE_DIR = Path("/root/wav2vec_unsupervised/data_preparation")
RAW_DIR = BASE_DIR / "raw_data"
AUDIO_TRAIN = BASE_DIR / "audio" / "train"
AUDIO_VAL = BASE_DIR / "audio" / "val"
AUDIO_TEST = BASE_DIR / "audio" / "test"
TEXT_DIR = BASE_DIR / "text"

# The specific dataset folders
DATA_FOLDERS = ["fisd-ga-90p", "fisd-ga-10p"]

# Create target directories
for d in [AUDIO_TRAIN, AUDIO_VAL, AUDIO_TEST, TEXT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

def clean_text(text):
    """Removes punctuation and converts to lowercase for the GAN."""
    text = text.lower()
    return text.translate(str.maketrans('', '', string.punctuation))

def process_dataset():
    print("Starting Ga dataset preparation pipeline...")
    
    all_ogg_files = []
    all_transcripts = set() 

    # 1. Process each specific dataset folder
    for folder_name in DATA_FOLDERS:
        folder_path = RAW_DIR / folder_name
        
        if not folder_path.exists():
            print(f"Warning: {folder_path} not found. Skipping.")
            continue

        # Gather audio files
        audios_dir = folder_path / "audios"
        if audios_dir.exists():
            for file in os.listdir(audios_dir):
                if file.endswith(".ogg"):
                    all_ogg_files.append(audios_dir / file)
        
        # Gather text dynamically
        csv_path = folder_path / "data.csv"
        if csv_path.exists():
            with open(csv_path, 'r', encoding='utf-8') as f:
                # Use Sniffer to automatically detect tabs vs commas
                sample = f.read(2048)
                f.seek(0)
                dialect = csv.Sniffer().sniff(sample)
                reader = csv.reader(f, dialect)
                
                header = next(reader, None)
                
                # Dynamically find the Transcription column index
                try:
                    clean_header = [h.strip() for h in header]
                    trans_idx = clean_header.index('Transcription')
                except ValueError:
                    # Fallback to index 2 based on your sample if the header name is weird
                    trans_idx = 2
                    
                for row in reader:
                    if len(row) > trans_idx:
                        ga_text = clean_text(row[trans_idx])
                        if ga_text.strip():
                            all_transcripts.add(ga_text)

    # 2. Write the unlabelled text corpus
    unlabelled_path = TEXT_DIR / "unlabelled.txt"
    with open(unlabelled_path, 'w', encoding='utf-8') as f:
        for text in all_transcripts:
            f.write(text + "\n")
    print(f"\n=> Saved {len(all_transcripts)} unique sentences to {unlabelled_path}")

    # 3. Shuffle and split the audio files (80/10/10)
    random.seed(42) 
    random.shuffle(all_ogg_files)
    
    total_files = len(all_ogg_files)
    train_split = int(total_files * 0.8)
    val_split = int(total_files * 0.9)
    
    splits = {
        "Train": (all_ogg_files[:train_split], AUDIO_TRAIN),
        "Validation": (all_ogg_files[train_split:val_split], AUDIO_VAL),
        "Test": (all_ogg_files[val_split:], AUDIO_TEST)
    }

    print(f"=> Total audio files found: {total_files}\n")

    # 4. Convert and move files using ffmpeg WITH PROGRESS BAR
    for split_name, (files, dest_dir) in splits.items():
        # Wrap the files list in tqdm() to generate the progress bar
        for ogg_path in tqdm(files, desc=f"Converting {split_name} Audio", unit="file"):
            wav_filename = ogg_path.stem + ".wav"
            wav_path = dest_dir / wav_filename
            
            cmd = ["ffmpeg", "-y", "-i", str(ogg_path), "-ar", "16000", "-ac", "1", str(wav_path)]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    print("\nData preparation complete! Your audio is strictly formatted and ready for Fairseq.")

if __name__ == "__main__":
    process_dataset()