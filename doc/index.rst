.. DeepSpeech documentation master file, created by
   sphinx-quickstart on Thu Feb  2 21:20:39 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to DeepSpeech's documentation!
======================================

DeepSpeech is an open source Speech-To-Text engine, using a model trained by machine learning techniques based on `Baidu's Deep Speech research paper <https://arxiv.org/abs/1412.5567>`_. Project DeepSpeech uses Google's `TensorFlow <https://www.tensorflow.org/>`_ to make the implementation easier.

To install and use DeepSpeech all you have to do is:

.. code-block:: bash

   # Create and activate a virtualenv
   virtualenv -p python3 $HOME/tmp/deepspeech-venv/
   source $HOME/tmp/deepspeech-venv/bin/activate

   # Install DeepSpeech
   pip3 install deepspeech

   # Download pre-trained English model files
   curl -LO https://github.com/mozilla/DeepSpeech/releases/download/v0.7.0/deepspeech-0.7.0-models.pbmm
   curl -LO https://github.com/mozilla/DeepSpeech/releases/download/v0.7.0/deepspeech-0.7.0-models.scorer

   # Download example audio files
   curl -LO https://github.com/mozilla/DeepSpeech/releases/download/v0.7.0/audio-0.7.0.tar.gz
   tar xvf audio-0.7.0.tar.gz

   # Transcribe an audio file
   deepspeech --model deepspeech-0.7.0-models.pbmm --scorer deepspeech-0.7.0-models.scorer --audio audio/2830-3980-0043.wav

A pre-trained English model is available for use and can be downloaded following the instructions in :ref:`the usage docs <usage-docs>`. For the latest release, including pre-trained models and checkpoints, `see the GitHub releases page <https://github.com/mozilla/DeepSpeech/releases/latest>`_.

Quicker inference can be performed using a supported NVIDIA GPU on Linux. See the `release notes <https://github.com/mozilla/DeepSpeech/releases/latest>`_ to find which GPUs are supported. To run ``deepspeech`` on a GPU, install the GPU specific package:

.. code-block:: bash

   # Create and activate a virtualenv
   virtualenv -p python3 $HOME/tmp/deepspeech-gpu-venv/
   source $HOME/tmp/deepspeech-gpu-venv/bin/activate

   # Install DeepSpeech CUDA enabled package
   pip3 install deepspeech-gpu

   # Transcribe an audio file.
   deepspeech --model deepspeech-0.7.0-models.pbmm --scorer deepspeech-0.7.0-models.scorer --audio audio/2830-3980-0043.wav

Please ensure you have the required :ref:`CUDA dependencies <cuda-deps>`.

See the output of ``deepspeech -h`` for more information on the use of ``deepspeech``. (If you experience problems running ``deepspeech``, please check :ref:`required runtime dependencies <runtime-deps>`).

.. toctree::
   :maxdepth: 2
   :caption: Introduction

   USING

   TRAINING

   Decoder

.. toctree::
   :maxdepth: 2
   :caption: DeepSpeech Model

   DeepSpeech

   Geometry

   ParallelOptimization

.. toctree::
   :maxdepth: 2
   :caption: Enums and structs

   Error-Codes

   Structs

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   C-API

   DotNet-API

   Java-API

   NodeJS-API

   Python-API

.. toctree::
   :maxdepth: 2
   :caption: Examples

   C-Examples

   DotNet-Examples

   Java-Examples

   NodeJS-Examples

   Python-Examples

.. toctree::
   :maxdepth: 2
   :caption: Contributed examples

   DotNet-contrib-examples

   NodeJS-contrib-Examples

   Python-contrib-Examples

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
