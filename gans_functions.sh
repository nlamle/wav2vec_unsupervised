# # #!/bin/bash

# # # This script runs the GANS training of  unsupervised wav2vec pipeline

# # # Wav2Vec Unsupervised Pipeline Runner
# # # This script runs the entire fairseq wav2vec unsupervised pipeline
# # # with checkpointing to allow resuming from any step

# # set -e  # Exit on error
# # set -o pipefail  # Exit if any command in a pipe fails

# # source utils.sh

# # #=========================== GANS training and preparation ==============================
# # train_gans(){
# #    local step_name="train_gans"
# #    export FAIRSEQ_ROOT=$FAIRSEQ_ROOT
# #    # export KALDI_ROOT="$DIR_PATH/pykaldi/tools/kaldi"
# #    export KENLM_ROOT="$KENLM_ROOT"
# #    export PYTHONPATH="/$DIR_PATH:$PYTHONPATH"


# #    if is_completed "$step_name"; then
# #         log "Skipping gans training  (already completed)"
# #         return 0
# #     fi

# #     log "gans training."
# #     mark_in_progress "$step_name"
   

# #    PYTHONPATH=$FAIRSEQ_ROOT PREFIX=w2v_unsup_gan_xp fairseq-hydra-train \
# #     -m --config-dir "$FAIRSEQ_ROOT/examples/wav2vec/unsupervised/config/gan" \
# #     --config-name w2vu \
# #     task.data="$CLUSTERING_DIR/precompute_pca512_cls128_mean_pooled" \
# #     task.text_data="$TEXT_OUTPUT/phones/" \
# #     task.kenlm_path="$TEXT_OUTPUT/phones/lm.phones.filtered.04.bin" \
# #     common.user_dir="$FAIRSEQ_ROOT/examples/wav2vec/unsupervised" \
# #     model.code_penalty=6,10 model.gradient_penalty=0.5,1.0 \
# #     model.smoothness_weight='1.5' 'common.seed=range(0,5)' \
# #     +optimizer.groups.generator.optimizer.lr="[0.00004]" \
# #     +optimizer.groups.discriminator.optimizer.lr="[0.00002]" \
# #     ~optimizer.groups.generator.optimizer.amsgrad \
# #     ~optimizer.groups.discriminator.optimizer.amsgrad \
# #     2>&1 | tee $RESULTS_DIR/training1.log

    

# #    if [ $? -eq 0 ]; then
# #         mark_completed "$step_name"
# #         log "gans trained successfully"
# #     else
# #         log "ERROR: gans training failed"
# #         exit 1
# #     fi
# # }



# #!/bin/bash

# # This script runs the GANS training of unsupervised wav2vec pipeline
# set -e  # Exit on error
# set -o pipefail  # Exit if any command in a pipe fails

# source utils.sh

# #=========================== GANS training and preparation ==============================
# train_gans(){
#    local step_name="train_gans"
#    export FAIRSEQ_ROOT=$FAIRSEQ_ROOT
#    export KENLM_ROOT="$KENLM_ROOT"
#    export PYTHONPATH="/$DIR_PATH:$PYTHONPATH"

#    if is_completed "$step_name"; then
#         log "Skipping gans training (already completed)"
#         return 0
#     fi

#     log "gans training."
#     mark_in_progress "$step_name"
   
#    # Note: Removed the -m (multirun) flag and simplified ranges to single values 
#    # to prevent crashing your 8GB Mac RAM.
#    PYTHONPATH=$FAIRSEQ_ROOT PREFIX=w2v_unsup_gan_xp fairseq-hydra-train \
#     --config-dir "$FAIRSEQ_ROOT/examples/wav2vec/unsupervised/config/gan" \
#     --config-name w2vu \
#     task.data="/root/wav2vec_unsupervised/data_preparation/english_audio" \
#     task.text_data="/root/wav2vec_unsupervised/data_preparation/english_text/unlabelled.txt" \
#     task.kenlm_path="" \
#     common.user_dir="$FAIRSEQ_ROOT/examples/wav2vec/unsupervised" \
#     model.code_penalty=6 \
#     model.gradient_penalty=0.5 \
#     model.smoothness_weight=1.5 \
#     common.seed=0 \
#     +optimizer.groups.generator.optimizer.lr="[0.0001]" \
#     +optimizer.groups.discriminator.optimizer.lr="[0.0001]" \
#     2>&1 | tee $RESULTS_DIR/training1.log

#    if [ $? -eq 0 ]; then
#         mark_completed "$step_name"
#         log "gans trained successfully"
#     else
#         log "ERROR: gans training failed"
#         exit 1
#     fi
# }

#!/bin/bash

# This script runs the GANS training of unsupervised wav2vec pipeline
set -e  # Exit on error
set -o pipefail  # Exit if any command in a pipe fails

source utils.sh

#=========================== GANS training and preparation ==============================
train_gans(){
   local step_name="train_gans"
   
   # Set up roots
   export FAIRSEQ_ROOT="/root/wav2vec_unsupervised/fairseq_"
   export KENLM_ROOT="$KENLM_ROOT"
   export PYTHONPATH="$FAIRSEQ_ROOT:$FAIRSEQ_ROOT/examples/wav2vec/unsupervised:$PYTHONPATH"

   if is_completed "$step_name"; then
        log "Skipping gans training (already completed)"
        return 0
    fi

    log "gans training."
    mark_in_progress "$step_name"
   
   # Note: We removed the +optimizer overrides. 
   # The values are already set in your w2vu.yaml.
   PREFIX=w2v_unsup_gan_xp fairseq-hydra-train \
    --config-dir "$FAIRSEQ_ROOT/examples/wav2vec/unsupervised/config/gan" \
    --config-name w2vu \
    common.user_dir="/root/wav2vec_unsupervised/fairseq_/examples/wav2vec/unsupervised" \
    task.data="/root/wav2vec_unsupervised/data_preparation/english_audio" \
    task.text_data="/root/wav2vec_unsupervised/data_preparation/english_text/unlabelled.txt" \
    common.seed=0 \
    2>&1 | tee $RESULTS_DIR/training1.log

   # Check exit status of the first command in the pipe (fairseq-hydra-train)
   if [ ${PIPESTATUS[0]} -eq 0 ]; then
        mark_completed "$step_name"
        log "gans trained successfully"
    else
        log "ERROR: gans training failed"
        exit 1
    fi
}