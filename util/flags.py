from __future__ import absolute_import, division, print_function

import os
import absl.flags

FLAGS = absl.flags.FLAGS

def create_flags():
    # Importer
    # ========

    f = absl.flags

    f.DEFINE_string('train_files', '', 'comma separated list of files specifying the dataset used for training. Multiple files will get merged. If empty, training will not be run.')
    f.DEFINE_string('train_files_sortby', 'wav_filesize', 'comma separated list, dataframe sortby target, add `-` to descending the target column, specify epoch with `:`, Ex: -transcript:3, transcript increase ~25% training time, None increase ~44%')
    f.DEFINE_string('dev_files', '', 'comma separated list of files specifying the dataset used for validation. Multiple files will get merged. If empty, validation will not be run.')
    f.DEFINE_string('test_files', '', 'comma separated list of files specifying the dataset used for testing. Multiple files will get merged. If empty, the model will not be tested.')

    f.DEFINE_string('feature_cache', '', 'cache MFCC features to disk to speed up future training runs ont he same data. This flag specifies the path where cached features extracted from --train_files will be saved. If empty, or if online augmentation flags are enabled, caching will be disabled.')

    f.DEFINE_integer('feature_win_len', 32, 'feature extraction audio window length in milliseconds')
    f.DEFINE_integer('feature_win_step', 20, 'feature extraction window step length in milliseconds')
    f.DEFINE_integer('audio_sample_rate', 16000, 'sample rate value expected by model')

    # Data Augmentation
    # ================

    f.DEFINE_string('audio_aug_mix_noise_walk_dirs', '', 'walk through wav dir, then mix noise wav into decoded audio')
    f.DEFINE_string('audio_aug_mix_noise_cache', '', 'must cache noise audio data, or it will read audio file every training step')
    f.DEFINE_float('audio_aug_mix_noise_max_noise_db', -25, 'to limit noise max volume')
    f.DEFINE_float('audio_aug_mix_noise_min_noise_db', -50, 'to limit noise min volume')
    f.DEFINE_float('audio_aug_mix_noise_max_audio_db', 0, 'to limit audio max volume')
    f.DEFINE_float('audio_aug_mix_noise_min_audio_db', -10, 'to limit audio min volume')

    f.DEFINE_float('data_aug_features_additive', 0, 'std of the Gaussian additive noise')
    f.DEFINE_float('data_aug_features_multiplicative', 0, 'std of normal distribution around 1 for multiplicative noise')

    f.DEFINE_float('augmentation_spec_dropout_keeprate', 1, 'keep rate of dropout augmentation on spectrogram (if 1, no dropout will be performed on spectrogram)')

    f.DEFINE_boolean('augmentation_sparse_warp', False, 'whether to use spectrogram sparse warp. USE OF THIS FLAG IS UNSUPPORTED, enable sparse warp will increase training time drastically, and the paper also mentioned that this is not a major factor to improve accuracy.')
    f.DEFINE_integer('augmentation_sparse_warp_num_control_points', 1, 'specify number of control points')
    f.DEFINE_integer('augmentation_sparse_warp_time_warping_para', 20, 'time_warping_para')
    f.DEFINE_integer('augmentation_sparse_warp_interpolation_order', 2, 'sparse_warp_interpolation_order')
    f.DEFINE_float('augmentation_sparse_warp_regularization_weight', 0.0, 'sparse_warp_regularization_weight')
    f.DEFINE_integer('augmentation_sparse_warp_num_boundary_points', 1, 'sparse_warp_num_boundary_points')

    f.DEFINE_boolean('augmentation_freq_and_time_masking', False, 'whether to use frequency and time masking augmentation')
    f.DEFINE_integer('augmentation_freq_and_time_masking_freq_mask_range', 5, 'max range of masks in the frequency domain when performing freqtime-mask augmentation')
    f.DEFINE_integer('augmentation_freq_and_time_masking_number_freq_masks', 3, 'number of masks in the frequency domain when performing freqtime-mask augmentation')
    f.DEFINE_integer('augmentation_freq_and_time_masking_time_mask_range', 2, 'max range of masks in the time domain when performing freqtime-mask augmentation')
    f.DEFINE_integer('augmentation_freq_and_time_masking_number_time_masks', 3, 'number of masks in the time domain when performing freqtime-mask augmentation')

    f.DEFINE_float('augmentation_speed_up_std', 0, 'std for speeding-up tempo. If std is 0, this augmentation is not performed')

    f.DEFINE_boolean('augmentation_pitch_and_tempo_scaling', False, 'whether to use spectrogram speed and tempo scaling')
    f.DEFINE_float('augmentation_pitch_and_tempo_scaling_min_pitch', 0.95, 'min value of pitch scaling')
    f.DEFINE_float('augmentation_pitch_and_tempo_scaling_max_pitch', 1.2, 'max value of pitch scaling')
    f.DEFINE_float('augmentation_pitch_and_tempo_scaling_max_tempo', 1.2, 'max vlaue of tempo scaling')

    f.DEFINE_boolean('ctc_loss_ignore_longer_outputs_than_inputs', False, 'augmentation might shrink time steps to cause ctc loss outputs longer than inputs, enable to prevent crash')

    # Global Constants
    # ================

    f.DEFINE_integer('epochs', 75, 'how many epochs (complete runs through the train files) to train for')

    f.DEFINE_float('dropout_rate', 0.05, 'dropout rate for feedforward layers')
    f.DEFINE_float('dropout_rate2', -1.0, 'dropout rate for layer 2 - defaults to dropout_rate')
    f.DEFINE_float('dropout_rate3', -1.0, 'dropout rate for layer 3 - defaults to dropout_rate')
    f.DEFINE_float('dropout_rate4', 0.0, 'dropout rate for layer 4 - defaults to 0.0')
    f.DEFINE_float('dropout_rate5', 0.0, 'dropout rate for layer 5 - defaults to 0.0')
    f.DEFINE_float('dropout_rate6', -1.0, 'dropout rate for layer 6 - defaults to dropout_rate')

    f.DEFINE_float('relu_clip', 20.0, 'ReLU clipping value for non-recurrent layers')

    # Adam optimizer(http://arxiv.org/abs/1412.6980) parameters

    f.DEFINE_float('beta1', 0.9, 'beta 1 parameter of Adam optimizer')
    f.DEFINE_float('beta2', 0.999, 'beta 2 parameter of Adam optimizer')
    f.DEFINE_float('epsilon', 1e-8, 'epsilon parameter of Adam optimizer')
    f.DEFINE_float('learning_rate', 0.001, 'learning rate of Adam optimizer')

    # Acoustic Length Normalization
    f.DEFINE_float("len_norm_logits_alpha", 0.0, "logits alpha", lower_bound=0.0, upper_bound=1.0)
    f.DEFINE_float("len_norm_logits_beta", 0.0, "logits beta", lower_bound=0.0)
    f.DEFINE_float("len_norm_transcript_alpha", 0.0, "transcript alpha", lower_bound=0.0, upper_bound=1.0)
    f.DEFINE_float("len_norm_transcript_beta", 0.0, "transcript beta", lower_bound=0.0)
    f.DEFINE_float("len_norm_avg_logits_length", 0.0, "for modifying loss, 0.0 means ignore", lower_bound=0.0)
    f.DEFINE_float("len_norm_avg_transcript_length", 0.0, "for modifying loss, 0.0 means ignore", lower_bound=0.0)

    # Batch sizes

    f.DEFINE_integer('train_batch_size', 1, 'number of elements in a training batch')
    f.DEFINE_integer('dev_batch_size', 1, 'number of elements in a validation batch')
    f.DEFINE_integer('test_batch_size', 1, 'number of elements in a test batch')

    f.DEFINE_integer('export_batch_size', 1, 'number of elements per batch on the exported graph')

    # Performance

    f.DEFINE_integer('inter_op_parallelism_threads', 0, 'number of inter-op parallelism threads - see tf.ConfigProto for more details. USE OF THIS FLAG IS UNSUPPORTED')
    f.DEFINE_integer('intra_op_parallelism_threads', 0, 'number of intra-op parallelism threads - see tf.ConfigProto for more details. USE OF THIS FLAG IS UNSUPPORTED')
    f.DEFINE_boolean('use_allow_growth', False, 'use Allow Growth flag which will allocate only required amount of GPU memory and prevent full allocation of available GPU memory')
    f.DEFINE_boolean('use_cudnn_rnn', False, 'use CuDNN RNN backend for training on GPU. Note that checkpoints created with this flag can only be used with CuDNN RNN, i.e. fine tuning on a CPU device will not work')
    f.DEFINE_string('cudnn_checkpoint', '', 'path to a checkpoint created using --use_cudnn_rnn. Specifying this flag allows one to convert a CuDNN RNN checkpoint to a checkpoint capable of running on a CPU graph.')

    f.DEFINE_boolean('automatic_mixed_precision', False, 'whether to allow automatic mixed precision training. USE OF THIS FLAG IS UNSUPPORTED. Checkpoints created with automatic mixed precision training will not be usable without mixed precision.')

    # Sample limits

    f.DEFINE_integer('limit_train', 0, 'maximum number of elements to use from train set - 0 means no limit')
    f.DEFINE_integer('limit_dev', 0, 'maximum number of elements to use from validation set- 0 means no limit')
    f.DEFINE_integer('limit_test', 0, 'maximum number of elements to use from test set- 0 means no limit')

    # Checkpointing

    f.DEFINE_string('checkpoint_dir', '', 'directory in which checkpoints are stored - defaults to directory "deepspeech/checkpoints" within user\'s data home specified by the XDG Base Directory Specification')
    f.DEFINE_integer('checkpoint_secs', 600, 'checkpoint saving interval in seconds')
    f.DEFINE_integer('max_to_keep', 5, 'number of checkpoint files to keep - default value is 5')
    f.DEFINE_string('load', 'auto', '"last" for loading most recent epoch checkpoint, "best" for loading best validated checkpoint, "init" for initializing a fresh model, "auto" for trying the other options in order last > best > init')

    # Exporting

    f.DEFINE_string('export_dir', '', 'directory in which exported models are stored - if omitted, the model won\'t get exported')
    f.DEFINE_boolean('remove_export', False, 'whether to remove old exported models')
    f.DEFINE_boolean('export_tflite', False, 'export a graph ready for TF Lite engine')
    f.DEFINE_integer('n_steps', 16, 'how many timesteps to process at once by the export graph, higher values mean more latency')
    f.DEFINE_string('export_language', '', 'language the model was trained on e.g. "en" or "English". Gets embedded into exported model.')
    f.DEFINE_boolean('export_zip', False, 'export a TFLite model and package with LM and info.json')
    f.DEFINE_string('export_name', 'output_graph', 'name for the export model')

    # Reporting

    f.DEFINE_integer('log_level', 1, 'log level for console logs - 0: INFO, 1: WARN, 2: ERROR, 3: FATAL')
    f.DEFINE_boolean('show_progressbar', True, 'Show progress for training, validation and testing processes. Log level should be > 0.')

    f.DEFINE_boolean('log_placement', False, 'whether to log device placement of the operators to the console')
    f.DEFINE_integer('report_count', 10, 'number of phrases with lowest WER(best matching) to print out during a WER report')

    f.DEFINE_string('summary_dir', '', 'target directory for TensorBoard summaries - defaults to directory "deepspeech/summaries" within user\'s data home specified by the XDG Base Directory Specification')

    f.DEFINE_string('test_output_file', '', 'path to a file to save all src/decoded/distance/loss tuples generated during a test epoch')

    # Geometry

    f.DEFINE_integer('n_hidden', 2048, 'layer width to use when initialising layers')

    # Initialization

    f.DEFINE_integer('random_seed', 4568, 'default random seed that is used to initialize variables')

    # Early Stopping

    f.DEFINE_boolean('early_stop', True, 'enable early stopping mechanism over validation dataset. If validation is not being run, early stopping is disabled.')
    f.DEFINE_integer('es_steps', 4, 'number of validations to consider for early stopping. Loss is not stored in the checkpoint so when checkpoint is revived it starts the loss calculation from start at that point')
    f.DEFINE_float('es_mean_th', 0.5, 'mean threshold for loss to determine the condition if early stopping is required')
    f.DEFINE_float('es_std_th', 0.5, 'standard deviation threshold for loss to determine the condition if early stopping is required')

    # Decoder

    f.DEFINE_boolean('utf8', False, 'enable UTF-8 mode. When this is used the model outputs UTF-8 sequences directly rather than using an alphabet mapping.')
    f.DEFINE_string('alphabet_config_path', 'data/alphabet.txt', 'path to the configuration file specifying the alphabet used by the network. See the comment in data/alphabet.txt for a description of the format.')
    f.DEFINE_string('lm_binary_path', 'data/lm/lm.binary', 'path to the language model binary file created with KenLM')
    f.DEFINE_alias('lm', 'lm_binary_path')
    f.DEFINE_string('lm_trie_path', 'data/lm/trie', 'path to the language model trie file created with native_client/generate_trie')
    f.DEFINE_alias('trie', 'lm_trie_path')
    f.DEFINE_integer('beam_width', 1024, 'beam width used in the CTC decoder when building candidate transcriptions')
    f.DEFINE_float('lm_alpha', 0.75, 'the alpha hyperparameter of the CTC decoder. Language Model weight.')
    f.DEFINE_float('lm_beta', 1.85, 'the beta hyperparameter of the CTC decoder. Word insertion weight.')
    f.DEFINE_float('cutoff_prob', 1.0, 'only consider characters until this probability mass is reached. 1.0 = disabled.')
    f.DEFINE_integer('cutoff_top_n', 300, 'only process this number of characters sorted by probability mass for each time step. If bigger than alphabet size, disabled.')

    # Fine Tune Decoder: tune alpha -> tune beta -> tune alpha

    f.DEFINE_list('finetune_lm_csv_files', '', 'comma separated list of files specifing the dataset used for tuning.')
    f.DEFINE_integer('finetune_lm_sample_size', 256, 'sampling audio from `finetune_lm_csv_files` to evaluate wer', lower_bound=1)
    f.DEFINE_float('finetune_lm_alpha_min', 0.5, 'constraint the alpha minimum value', lower_bound=0.01)
    f.DEFINE_float('finetune_lm_alpha_max', 1.5, 'constraint the alpha maximum value', lower_bound=0.01)
    f.DEFINE_integer('finetune_lm_alpha_steps', 5, 'scan alpha from finetune_lm_alpha_min to finetune_lm_alpha_max by steps.', lower_bound=3)
    f.DEFINE_float('finetune_lm_beta_min', 0.5, 'constraint the beta minimum value', lower_bound=0.01)
    f.DEFINE_float('finetune_lm_beta_max', 1.5, 'constraint the beta maximum value', lower_bound=0.01)
    f.DEFINE_integer('finetune_lm_beta_steps', 3, 'scan alpha from finetune_lm_beta_min to finetune_lm_beta_max by steps', lower_bound=3)
    f.DEFINE_float('finetune_lm_alpha_radius', 0.05, 'set alpha radius for "random_search".', lower_bound=0.0001)
    f.DEFINE_float('finetune_lm_beta_radius', 0.1, 'set beta radius for "random_search".', lower_bound=0.0001)
    f.DEFINE_integer('finetune_lm_n_iterations', 10, 'set iterations for "random_search".', lower_bound=1)
    f.DEFINE_enum('finetune_lm_target_mean_loss', 'word_distance', ['word_distance', 'char_distance', 'wer', 'cer'], 'specify target mean loss.')
    f.DEFINE_enum('finetune_lm_method', 'random_search', ['parabola', 'random_search', 'parabola+random_search'], 'specify finetune method.')
    f.DEFINE_string('finetune_lm_output_file', '', 'path to a file to save best alphabet and beta')

    # Inference mode

    f.DEFINE_string('one_shot_infer', '', 'one-shot inference mode: specify a wav file and the script will load the checkpoint and perform inference on it.')

    # Register validators for paths which require a file to be specified

    f.register_validator('alphabet_config_path',
                         os.path.isfile,
                         message='The file pointed to by --alphabet_config_path must exist and be readable.')

    f.register_validator('one_shot_infer',
                         lambda value: not value or os.path.isfile(value),
                         message='The file pointed to by --one_shot_infer must exist and be readable.')

    f.register_validator('finetune_lm_csv_files',
                         lambda filenames: not filenames or all([os.path.isfile(filename) for filename in filenames]),
                         message='The all --finetune_lm_csv_files must exist and be readable')
