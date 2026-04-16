#!/bin/bash

# Defines evaluation functions for the Wav2Vec-U pipeline.
# Source this file to load functions.

set -e            # Exit on error
set -o pipefail   # Exit if any command in a pipe fails

source utils.sh   # Load shared paths and helper functions (DIR_PATH, FAIRSEQ_ROOT, etc.)

# Shared setup for English LibriSpeech.
AUDIO_DIR="$DIR_PATH/data_preparation/english_audio"
TEXT_DIR="$DIR_PATH/data_preparation/english_text"
CHECKPOINT="$DIR_PATH/checkpoints/english_clean_5k/checkpoint_best.pt"
RESULTS_DIR="$DIR_PATH/data/results/librispeech"

# Helper Functions

create_dirs() {
    mkdir -p "$MANIFEST_DIR" "$CLUSTERING_DIR" "$MANIFEST_NONSIL_DIR" \
             "$RESULTS_DIR" "$CHECKPOINT_DIR" "$LOG_DIR" "$TEXT_OUTPUT" \
             "$GANS_OUTPUT_PHONES" "$DIR_PATH/inference_results"
}

# ============================ MODEL EVALUATION =======================
# transcription_gans_viterbi: Runs Viterbi decoding on the test set.
# Outputs phonetic transcriptions to the $GANS_OUTPUT_PHONES directory.
# Optional arg: $1 = path to a specific checkpoint (defaults to $CHECKPOINT above)
transcription_gans_viterbi() {
    local model_path="${1:-$CHECKPOINT}"

    if [ ! -f "$model_path" ]; then
        echo "[ERROR] Checkpoint not found at: $model_path"
        echo "  Please set the CHECKPOINT variable or pass a path as an argument."
        exit 1
    fi

    echo "[INFO] Running Viterbi decoding..."
    echo "  Checkpoint : $model_path"
    echo "  Audio data : $AUDIO_DIR"
    echo "  Text data  : $TEXT_DIR/unlabelled.txt"
    echo "  Results dir: $GANS_OUTPUT_PHONES"

    export HYDRA_FULL_ERROR=1
    export FAIRSEQ_ROOT="$DIR_PATH/fairseq_"
    export PYTHONPATH="$FAIRSEQ_ROOT:$FAIRSEQ_ROOT/examples/wav2vec/unsupervised:${PYTHONPATH:-}"

    python "$FAIRSEQ_ROOT/examples/wav2vec/unsupervised/w2vu_generate.py" \
        --config-dir "$FAIRSEQ_ROOT/examples/wav2vec/unsupervised/config/generate" \
        --config-name viterbi \
        fairseq.common.user_dir="$FAIRSEQ_ROOT/examples/wav2vec/unsupervised" \
        fairseq.task._name="unsupervised_speech" \
        fairseq.task.data="$DIR_PATH/data_preparation/english_audio" \
        fairseq.task.text_data="$TEXT_DIR/unlabelled.txt" \
        fairseq.common_eval.path="$model_path" \
        fairseq.dataset.gen_subset=test \
        fairseq.dataset.batch_size=1 \
        fairseq.dataset.num_workers=0 \
        fairseq.dataset.required_batch_size_multiple=1 \
        results_path="$GANS_OUTPUT_PHONES"

    echo "[INFO] Decoding complete. Results saved to: $GANS_OUTPUT_PHONES"
}
