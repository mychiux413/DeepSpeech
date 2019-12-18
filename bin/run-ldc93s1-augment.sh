#!/bin/sh
set -xe
if [ ! -f DeepSpeech.py ]; then
    echo "Please make sure you run this from DeepSpeech's top level directory."
    exit 1
fi;

if [ ! -f "data/ldc93s1/ldc93s1.csv" ]; then
    echo "Downloading and preprocessing LDC93S1 example data, saving in ./data/ldc93s1."
    python -u bin/import_ldc93s1.py ./data/ldc93s1
fi;

if [ -d "${COMPUTE_KEEP_DIR}" ]; then
    checkpoint_dir=$COMPUTE_KEEP_DIR
else
    checkpoint_dir=$(python -c 'from xdg import BaseDirectory as xdg; print(xdg.save_data_path("deepspeech/ldc93s1"))')
fi

# Force only one visible device because we have a single-sample dataset
# and when trying to run on multiple devices (like GPUs), this will break
export CUDA_VISIBLE_DEVICES=0

# rm -rf ~/.local/share/deepspeech
python -u DeepSpeech.py --noshow_progressbar \
  --train_files data/ldc93s1/ldc93s1.csv \
  --dev_files data/ldc93s1/ldc93s1.csv \
  --test_files data/ldc93s1/ldc93s1.csv \
  --train_batch_size 1 \
  --dev_batch_size 1 \
  --test_batch_size 1 \
  --n_hidden 100 \
  --epochs 200 \
  --checkpoint_dir "$checkpoint_dir" \
  --augmentation_sparse_warp true \
  --augmentation_sparse_warp_prob 1.0 \
  --augmentation_sparse_warp_time_warping_para 14 \
  --augmentation_sparse_warp_num_control_points 2 \
  --augmentation_pitch_and_tempo_scaling true \
  --augmentation_pitch_and_tempo_scaling_prob 1.0 \
  --augmentation_pitch_and_tempo_scaling_max_tempo 1.4 \
  --augmentation_pitch_and_tempo_scaling_min_pitch 0.8 \
  --augmentation_pitch_and_tempo_scaling_min_pitch 1.2 \
  --augmentation_freq_and_time_masking true \
  --augmentation_freq_and_time_masking_prob 1.0 \
  --augmentation_freq_and_time_masking_freq_mask_range 20 \
  --augmentation_freq_and_time_masking_time_mask_range 5 \
  --augmentation_speed_up_std 0.1 \
  --augmentation_speed_up_prob 1.0 \
  --data_aug_features_additive 0.1 \
  --data_aug_features_additive_prob 1.0 \
  --data_aug_features_multiplicative 0.1 \
  --data_aug_features_multiplicative_prob 1.0 \
  --augmentation_spec_dropout_keeprate 0.95 \
  --augmentation_spec_dropout_keeprate_prob 1.0 \
  --noearly_stop \
  --dropout_rate 0.2 \
  --learning_rate 0.0001 \
  "$@"