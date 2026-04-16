# Unsupervised Wav2Vec Pipeline

A suite of scripts to automate the Fairseq Wav2Vec 2.0 Unsupervised Speech Recognition pipeline. This implementation is designed for phonetic transcription and GAN-based adversarial training.

## System Requirements

- **OS**: Linux-based (recommended)
- **GPU**: NVIDIA GPU with CUDA 12.3 support
- **Python**: 3.10+ in a virtual environment
- **Tools**: Git, Docker (optional, but used in current deployment)

## Getting Started

### 1. Make Scripts Executable
```bash
chmod +x setup_functions.sh eval_functions.sh gans_functions.sh run_setup.sh utils.sh
```

### 2. Environment Setup
Install dependencies and configure the Fairseq environment:
```bash
./run_setup.sh
```

### 3. Data Preparation
Prepare audio and text datasets for unsupervised training:
```bash
./run_wav2vec.sh "/path/to/train_audio" \
                "/path/to/val_audio" \
                "/path/to/test_audio" \
                "/path/to/text_unlabelled"
```

### 4. GAN Training
Adjust hyperparameters in `fairseq_/examples/wav2vec/unsupervised/config/gan/w2vu.yaml` if needed, then launch training:
```bash
./run_gans.sh
```

### 5. Evaluation
Run Viterbi decoding using a trained checkpoint:
```bash
./run_eval.sh "/path/to/checkpoint.pt"
```
Phonetic transcriptions are saved to `data/transcription_phones/test.txt`.

## Summary
1. Ensure CUDA 12.3 is installed.
2. Initialize environment with `run_setup.sh`.
3. Process data with `run_wav2vec.sh`.
4. Train GAN with `run_gans.sh`.
5. Generate transcriptions with `run_eval.sh`.
