# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

import os

from functools import partial

import numpy as np
import pandas
import tensorflow as tf

from tensorflow.python.ops import gen_audio_ops as contrib_audio

from util.config import Config
from util.text import text_to_char_array
from util.flags import FLAGS
from util.spectrogram_augmentations import augment_freq_time_mask, augment_dropout, augment_pitch_and_tempo, augment_speed_up, augment_sparse_warp
from util.audio import read_frames_from_file, vad_split, DEFAULT_FORMAT
from util.audio_augmentation import augment_noise, create_noise_iterator, gla


def read_csvs(csv_files):
    sets = []
    for csv in csv_files:
        file = pandas.read_csv(csv, encoding='utf-8', na_filter=False)
        #FIXME: not cross-platform
        csv_dir = os.path.dirname(os.path.abspath(csv))
        file['wav_filename'] = file['wav_filename'].str.replace(r'(^[^/])', lambda m: os.path.join(csv_dir, m.group(1))) # pylint: disable=cell-var-from-loop
        sets.append(file)
    # Concat all sets, drop any extra columns, re-index the final result as 0..N
    return pandas.concat(sets, join='inner', ignore_index=True)


def samples_to_mfccs(samples, sample_rate, train_phase=False):
    spectrogram = contrib_audio.audio_spectrogram(samples,
                                                  window_size=Config.audio_window_samples,
                                                  stride=Config.audio_step_samples,
                                                  magnitude_squared=True)

    # Data Augmentations
    if train_phase:

        if FLAGS.augmentation_pitch_and_tempo_scaling:
            spectrogram = augment_pitch_and_tempo(spectrogram,
                                                  max_tempo=FLAGS.augmentation_pitch_and_tempo_scaling_max_tempo,
                                                  max_pitch=FLAGS.augmentation_pitch_and_tempo_scaling_max_pitch,
                                                  min_pitch=FLAGS.augmentation_pitch_and_tempo_scaling_min_pitch)

        if FLAGS.augmentation_speed_up_std > 0:
            spectrogram = augment_speed_up(spectrogram,
                                           speed_std=FLAGS.augmentation_speed_up_std)

        # sparse warp must before freq/time masking
        if FLAGS.augmentation_sparse_warp:
            spectrogram = augment_sparse_warp(spectrogram,
                                              time_warping_para=FLAGS.augmentation_sparse_warp_time_warping_para,
                                              interpolation_order=FLAGS.augmentation_sparse_warp_interpolation_order,
                                              regularization_weight=FLAGS.augmentation_sparse_warp_regularization_weight,
                                              num_boundary_points=FLAGS.augmentation_sparse_warp_num_boundary_points,
                                              num_control_points=FLAGS.augmentation_sparse_warp_num_control_points)

        if FLAGS.augmentation_freq_and_time_masking:
            spectrogram = augment_freq_time_mask(spectrogram,
                                                 frequency_masking_para=FLAGS.augmentation_freq_and_time_masking_freq_mask_range,
                                                 time_masking_para=FLAGS.augmentation_freq_and_time_masking_time_mask_range,
                                                 frequency_mask_num=FLAGS.augmentation_freq_and_time_masking_number_freq_masks,
                                                 time_mask_num=FLAGS.augmentation_freq_and_time_masking_number_time_masks)

        if FLAGS.augmentation_spec_dropout_keeprate < 1:
            spectrogram = augment_dropout(spectrogram,
                                          keep_prob=FLAGS.augmentation_spec_dropout_keeprate)

    mfccs = contrib_audio.mfcc(spectrogram=spectrogram,
                               sample_rate=sample_rate,
                               dct_coefficient_count=Config.n_input,
                               upper_frequency_limit=FLAGS.audio_sample_rate/2)
    mfccs = tf.reshape(mfccs, [-1, Config.n_input])

    review_audio = samples
    if FLAGS.review_audio_steps and train_phase and any([
                FLAGS.augmentation_spec_dropout_keeprate < 1,
                FLAGS.augmentation_freq_and_time_masking,
                FLAGS.augmentation_pitch_and_tempo_scaling,
                FLAGS.augmentation_speed_up_std > 0]):
        review_audio = gla(spectrogram, FLAGS.review_audio_gla_iterations)

    return mfccs, tf.shape(input=mfccs)[0], review_audio


def audiofile_to_features(wav_filename, train_phase=False, noise_iterator=None, speech_iterator=None):
    samples = tf.io.read_file(wav_filename)
    decoded = contrib_audio.decode_wav(samples, desired_channels=1)
    audio = decoded.audio

    # augment audio
    if noise_iterator or speech_iterator:
        audio = augment_noise(
            audio,
            noise_iterator,
            speech_iterator,
            min_n_noises=FLAGS.audio_aug_min_n_noises,
            max_n_noises=FLAGS.audio_aug_max_n_noises,
            min_n_speakers=FLAGS.audio_aug_min_n_speakers,
            max_n_speakers=FLAGS.audio_aug_max_n_speakers,
            min_audio_dbfs=FLAGS.audio_aug_min_audio_dbfs,
            max_audio_dbfs=FLAGS.audio_aug_max_audio_dbfs,
            min_noise_snr_db=FLAGS.audio_aug_min_noise_snr_db,
            max_noise_snr_db=FLAGS.audio_aug_max_noise_snr_db,
            min_speech_snr_db=FLAGS.audio_aug_min_speech_snr_db,
            max_speech_snr_db=FLAGS.audio_aug_max_speech_snr_db,
            limit_audio_peak_dbfs=FLAGS.audio_aug_limit_audio_peak_dbfs,
            limit_noise_peak_dbfs=FLAGS.audio_aug_limit_noise_peak_dbfs,
            limit_speech_peak_dbfs=FLAGS.audio_aug_limit_speech_peak_dbfs,
            sample_rate=FLAGS.audio_sample_rate,
        )

    features, features_len, review_audio = samples_to_mfccs(audio, decoded.sample_rate, train_phase=train_phase)

    # augment features
    if train_phase:
        if FLAGS.data_aug_features_multiplicative > 0:
            features *= tf.random.normal(
                mean=1, stddev=FLAGS.data_aug_features_multiplicative, shape=tf.shape(features))

        if FLAGS.data_aug_features_additive > 0:
            features += tf.random.normal(
                mean=0.0, stddev=FLAGS.data_aug_features_additive, shape=tf.shape(features))

    return features, features_len, review_audio


def entry_to_features(wav_filename, transcript, train_phase, noise_iterator, speech_iterator):
    # https://bugs.python.org/issue32117
    features, features_len, review_audio = audiofile_to_features(wav_filename, train_phase=train_phase, noise_iterator=noise_iterator, speech_iterator=speech_iterator)
    return wav_filename, features, features_len, tf.SparseTensor(*transcript), review_audio


def to_sparse_tuple(sequence):
    r"""Creates a sparse representention of ``sequence``.
        Returns a tuple with (indices, values, shape)
    """
    indices = np.asarray(list(zip([0]*len(sequence), range(len(sequence)))), dtype=np.int64)
    shape = np.asarray([1, len(sequence)], dtype=np.int64)
    return indices, sequence, shape


def create_dataset(csvs, batch_size, enable_cache=False, cache_path=None, train_phase=False, sortby='wav_filesize', noise_sources=None, speech_sources=None):
    df = read_csvs(csvs)
    if sortby:
        ascending = True
        if sortby.startswith('-'):
            ascending = False
            sortby = sortby[1:]
        assert sortby in df
        df.sort_values(by=sortby, inplace=True, ascending=ascending)

        if sortby == 'transcript':
            df['transcript_len'] = df['transcript'].apply(len)
            df.sort_values(by='transcript_len', inplace=True, ascending=ascending)

    df['transcript'] = df.apply(text_to_char_array, alphabet=Config.alphabet, result_type='reduce', axis=1)

    def generate_values():
        for _, row in df.iterrows():
            yield row.wav_filename, to_sparse_tuple(row.transcript)

    # Batching a dataset of 2D SparseTensors creates 3D batches, which fail
    # when passed to tf.nn.ctc_loss, so we reshape them to remove the extra
    # dimension here.
    def sparse_reshape(sparse):
        shape = sparse.dense_shape
        return tf.sparse.reshape(sparse, [shape[0], shape[2]])

    def batch_fn(wav_filenames, features, features_len, transcripts, review_audios):
        features = tf.data.Dataset.zip((features, features_len))
        features = features.padded_batch(batch_size,
                                         padded_shapes=([None, Config.n_input], []))
        transcripts = transcripts.batch(batch_size).map(sparse_reshape)
        wav_filenames = wav_filenames.batch(batch_size)

        # In order not to waste too much prefetch performance, randomly extract only `one` audio for each step
        if FLAGS.review_audio_steps and batch_size > 1:
            skip_size = tf.random.uniform(shape=[], minval=0, maxval=batch_size - 1, dtype=tf.int64)
            review_audio = review_audios.skip(skip_size).batch(1)
        else:
            review_audio = review_audios.batch(1)

        return tf.data.Dataset.zip((wav_filenames, features, transcripts, review_audio))

    num_gpus = len(Config.available_devices)

    noise_iterator = create_noise_iterator(noise_sources, read_csvs) if noise_sources else None
    speech_iterator = create_noise_iterator(speech_sources, read_csvs) if speech_sources else None


    process_fn = partial(entry_to_features, train_phase=train_phase, noise_iterator=noise_iterator, speech_iterator=speech_iterator)

    dataset = tf.data.Dataset.from_generator(generate_values,
                                             output_types=(tf.string, (tf.int64, tf.int32, tf.int64)))
    if not sortby:
        dataset = dataset.shuffle(min(len(df), 102400))

    dataset = dataset.map(process_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    if enable_cache:
        dataset = dataset.cache(cache_path)

    dataset = (dataset.window(batch_size, drop_remainder=True).flat_map(batch_fn)
                      .prefetch(num_gpus))

    return dataset


def split_audio_file(audio_path,
                     audio_format=DEFAULT_FORMAT,
                     batch_size=1,
                     aggressiveness=3,
                     outlier_duration_ms=10000,
                     outlier_batch_size=1):
    sample_rate, _, sample_width = audio_format
    multiplier = 1.0 / (1 << (8 * sample_width - 1))

    def generate_values():
        frames = read_frames_from_file(audio_path)
        segments = vad_split(frames, aggressiveness=aggressiveness)
        for segment in segments:
            segment_buffer, time_start, time_end = segment
            samples = np.frombuffer(segment_buffer, dtype=np.int16)
            samples = samples * multiplier
            samples = np.expand_dims(samples, axis=1)
            yield time_start, time_end, samples

    def to_mfccs(time_start, time_end, samples):
        features, features_len, _ = samples_to_mfccs(samples, sample_rate)
        return time_start, time_end, features, features_len

    def create_batch_set(bs, criteria):
        return (tf.data.Dataset
                .from_generator(generate_values, output_types=(tf.int32, tf.int32, tf.float32))
                .map(to_mfccs, num_parallel_calls=tf.data.experimental.AUTOTUNE)
                .filter(criteria)
                .padded_batch(bs, padded_shapes=([], [], [None, Config.n_input], [])))

    nds = create_batch_set(batch_size,
                           lambda start, end, f, fl: end - start <= int(outlier_duration_ms))
    ods = create_batch_set(outlier_batch_size,
                           lambda start, end, f, fl: end - start > int(outlier_duration_ms))
    dataset = nds.concatenate(ods)
    dataset = dataset.prefetch(len(Config.available_devices))
    return dataset


def secs_to_hours(secs):
    hours, remainder = divmod(secs, 3600)
    minutes, seconds = divmod(remainder, 60)
    return '%d:%02d:%02d' % (hours, minutes, seconds)
