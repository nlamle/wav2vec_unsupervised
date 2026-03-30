import os
import subprocess
from pathlib import Path
from tqdm import tqdm

# Paths for English setup
BASE_DIR = Path("/root/wav2vec_unsupervised/data_preparation")
RAW_ENGLISH = BASE_DIR / "raw_data/english_mini/LibriSpeech/dev-clean"
AUDIO_OUT = BASE_DIR / "english_audio"
TEXT_OUT = BASE_DIR / "english_text"

# Create directories
AUDIO_OUT.mkdir(parents=True, exist_ok=True)
TEXT_OUT.mkdir(parents=True, exist_ok=True)

def process_english():
    print("Converting LibriSpeech FLAC to 16kHz WAV...")
    
    # Find all .flac files in the nested LibriSpeech structure
    flac_files = list(RAW_ENGLISH.glob("**/*.flac"))
    
    # 1. Convert Audio
    for flac_path in tqdm(flac_files, desc="Converting Audio", unit="file"):
        wav_path = AUDIO_OUT / (flac_path.stem + ".wav")
        # Convert to 16kHz, Mono WAV
        cmd = ["ffmpeg", "-y", "-i", str(flac_path), "-ar", "16000", "-ac", "1", str(wav_path)]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # 2. Extract Text (LibriSpeech provides .trans.txt files in each folder)
    print("Extracting English transcriptions...")
    all_text = []
    trans_files = list(RAW_ENGLISH.glob("**/*.trans.txt"))
    
    for trans_file in trans_files:
        with open(trans_file, 'r') as f:
            for line in f:
                # LibriSpeech format: "ID-ID-SEQ TEXT CONTENT"
                # We just want the TEXT CONTENT
                parts = line.strip().split(' ', 1)
                if len(parts) > 1:
                    all_text.append(parts[1].lower())

    with open(TEXT_OUT / "unlabelled.txt", "w") as f:
        for line in set(all_text): # set() removes duplicates
            f.write(line + "\n")

    print(f"Done! Saved {len(flac_files)} WAVs and {len(all_text)} unique English sentences.")

if __name__ == "__main__":
    process_english()