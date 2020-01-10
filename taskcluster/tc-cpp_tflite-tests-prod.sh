#!/bin/bash

set -xe

#TODO: Remove after 0.6.1
export DEEPSPEECH_PROD_MODEL=https://github.com/lissyx/DeepSpeech/releases/download/v0.6.0/output_graph.tflite
export DEEPSPEECH_PROD_MODEL_MMAP=https://github.com/lissyx/DeepSpeech/releases/download/v0.6.0/output_graph.tflite

source $(dirname "$0")/tc-tests-utils.sh

model_source=${DEEPSPEECH_PROD_MODEL//.pb/.tflite}
model_name=$(basename "${model_source}")
model_name_mmap=$(basename "${model_source}")
model_source_mmap=${DEEPSPEECH_PROD_MODEL_MMAP//.pbmm/.tflite}
export DEEPSPEECH_ARTIFACTS_ROOT=${DEEPSPEECH_ARTIFACTS_TFLITE_ROOT}
export DATA_TMP_DIR=${TASKCLUSTER_TMP_DIR}

download_material "${TASKCLUSTER_TMP_DIR}/ds"

export PATH=${TASKCLUSTER_TMP_DIR}/ds/:$PATH

check_tensorflow_version

run_prodtflite_inference_tests
