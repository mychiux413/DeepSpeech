import tensorflow as tf
import tensorflow.compat.v1 as tfv1
from util.sparse_image_warp import sparse_image_warp
import numpy as np


def augment_freq_time_mask(spectrogram,
                           frequency_masking_para=30,
                           time_masking_para=10,
                           frequency_mask_num=3,
                           time_mask_num=3,
                           prob=0.2):
    return tf.cond(tf.math.greater(tfv1.random_uniform([], 0.0, 1.0, tf.float32), prob),
                   lambda: spectrogram,
                   lambda: _do_freq_time_mask(
                       spectrogram, frequency_masking_para, time_masking_para, frequency_mask_num, time_mask_num)
                   )


def augment_pitch_and_tempo(spectrogram,
                            max_tempo=1.2,
                            max_pitch=1.1,
                            min_pitch=0.95,
                            prob=0.2):
    return tf.cond(tf.math.greater(tfv1.random_uniform([], 0.0, 1.0, tf.float32), prob),
                   lambda: spectrogram,
                   lambda: _do_pitch_and_tempo(
                       spectrogram, max_tempo, max_pitch, min_pitch)
                   )


def augment_speed_up(spectrogram,
                     speed_std=0.1, prob=0.2):
    return tf.cond(tf.math.greater(tfv1.random_uniform([], 0.0, 1.0, tf.float32), prob),
                   lambda: spectrogram,
                   lambda: _do_speed_up(spectrogram, speed_std)
                   )


def augment_dropout(spectrogram, keep_prob=0.95, prob=0.2):
    return tf.cond(tf.math.greater(tfv1.random_uniform([], 0.0, 1.0, tf.float32), prob),
                   lambda: spectrogram,
                   lambda: _do_augment_dropout(spectrogram, keep_prob)
                   )


def augment_sparse_warp(spectrogram, time_warping_para=80, interpolation_order=2, regularization_weight=0.0, num_boundary_points=1, num_control_points=1, prob=0.2):
    """Reference: https://arxiv.org/pdf/1904.08779.pdf
    Args:
        spectrogram: `[batch, time, frequency]` float `Tensor`
        time_warping_para: 'W' parameter in paper
        interpolation_order: used to put into `sparse_image_warp`
        regularization_weight: used to put into `sparse_image_warp`
        num_boundary_points: used to put into `sparse_image_warp`,
                            default=1 means boundary points on 4 corners of the image
        num_control_points: number of control points
    Returns:
        warped_spectrogram: `[batch, time, frequency]` float `Tensor` with same
            type as input image.
    """
    return tf.cond(tf.math.greater(np.random.uniform(0.0, 1.0), prob),
                   lambda: spectrogram,
                   lambda: _do_sparse_warp(spectrogram, time_warping_para, interpolation_order, regularization_weight,
                                           num_boundary_points, num_control_points)
                   )


def _do_augment_dropout(spectrogram,
                        keep_prob=0.95):
    return tf.nn.dropout(spectrogram, rate=1-keep_prob)


def _do_freq_time_mask(mel_spectrogram,
                       frequency_masking_para=30,
                       time_masking_para=10,
                       frequency_mask_num=3,
                       time_mask_num=3):
    time_max = tf.shape(mel_spectrogram)[1]
    freq_max = tf.shape(mel_spectrogram)[2]
    # Frequency masking
    for _ in range(frequency_mask_num):
        f = tf.random.uniform(
            shape=(), minval=0, maxval=frequency_masking_para, dtype=tf.dtypes.int32)
        f0 = tf.random.uniform(
            shape=(), minval=0, maxval=freq_max - f, dtype=tf.dtypes.int32)
        value_ones_freq_prev = tf.ones(shape=[1, time_max, f0])
        value_zeros_freq = tf.zeros(shape=[1, time_max, f])
        value_ones_freq_next = tf.ones(shape=[1, time_max, freq_max-(f0+f)])
        freq_mask = tf.concat(
            [value_ones_freq_prev, value_zeros_freq, value_ones_freq_next], axis=2)
        # mel_spectrogram[:, f0:f0 + f, :] = 0 #can't assign to tensor
        # mel_spectrogram[:, f0:f0 + f, :] = value_zeros_freq #can't assign to tensor
        mel_spectrogram = mel_spectrogram*freq_mask

    # Time masking
    for _ in range(time_mask_num):
        t = tf.random.uniform(shape=(), minval=0,
                              maxval=time_masking_para, dtype=tf.dtypes.int32)
        t0 = tf.random.uniform(
            shape=(), minval=0, maxval=time_max - t, dtype=tf.dtypes.int32)
        value_zeros_time_prev = tf.ones(shape=[1, t0, freq_max])
        value_zeros_time = tf.zeros(shape=[1, t, freq_max])
        value_zeros_time_next = tf.ones(shape=[1, time_max-(t0+t), freq_max])
        time_mask = tf.concat(
            [value_zeros_time_prev, value_zeros_time, value_zeros_time_next], axis=1)
        # mel_spectrogram[:, :, t0:t0 + t] = 0 #can't assign to tensor
        # mel_spectrogram[:, :, t0:t0 + t] = value_zeros_time #can't assign to tensor
        mel_spectrogram = mel_spectrogram*time_mask

    return mel_spectrogram


def _do_pitch_and_tempo(spectrogram,
                        max_tempo=1.2,
                        max_pitch=1.1,
                        min_pitch=0.95):
    original_shape = tf.shape(spectrogram)
    choosen_pitch = tf.random.uniform(
        shape=(), minval=min_pitch, maxval=max_pitch)
    choosen_tempo = tf.random.uniform(shape=(), minval=1, maxval=max_tempo)
    new_freq_size = tf.cast(
        tf.cast(original_shape[2], tf.float32)*choosen_pitch, tf.int32)
    new_time_size = tf.cast(
        tf.cast(original_shape[1], tf.float32)/(choosen_tempo), tf.int32)
    spectrogram_aug = tf.image.resize_bilinear(
        tf.expand_dims(spectrogram, -1), [new_time_size, new_freq_size])
    spectrogram_aug = tf.image.crop_to_bounding_box(spectrogram_aug, offset_height=0, offset_width=0, target_height=tf.shape(
        spectrogram_aug)[1], target_width=tf.minimum(original_shape[2], new_freq_size))
    spectrogram_aug = tf.cond(choosen_pitch < 1,
                              lambda: tf.image.pad_to_bounding_box(spectrogram_aug, offset_height=0, offset_width=0,
                                                                   target_height=tf.shape(spectrogram_aug)[1], target_width=original_shape[2]),
                              lambda: spectrogram_aug)
    return spectrogram_aug[:, :, :, 0]


def _do_speed_up(spectrogram,
                 speed_std=0.1):
    original_shape = tf.shape(spectrogram)
    # abs makes sure the augmention will only speed up
    choosen_speed = tf.math.abs(tf.random.normal(shape=(), stddev=speed_std))
    choosen_speed = 1 + choosen_speed
    new_freq_size = tf.cast(tf.cast(original_shape[2], tf.float32), tf.int32)
    new_time_size = tf.cast(
        tf.cast(original_shape[1], tf.float32)/(choosen_speed), tf.int32)
    spectrogram_aug = tf.image.resize_bilinear(
        tf.expand_dims(spectrogram, -1), [new_time_size, new_freq_size])
    return spectrogram_aug[:, :, :, 0]


def _do_sparse_warp(spectrogram, time_warping_para=80, interpolation_order=2, regularization_weight=0.0, num_boundary_points=1, num_control_points=1):
    # resize to fit `sparse_image_warp`'s input shape
    # (1, time steps, freq, 1), batch_size must be 1
    spectrogram = tf.expand_dims(spectrogram, -1)

    original_shape = tf.shape(spectrogram)
    tau, freq_size = original_shape[1], original_shape[2]

    # to protect short audio
    time_warping_para = tf.math.minimum(
        time_warping_para, tf.math.subtract(tf.math.floordiv(tau, 2), 1))

    choosen_freqs = tf.random.shuffle(tf.add(tf.range(freq_size), 1))[
        0: num_control_points]

    sources = []
    dests = []
    for i in range(num_control_points):
        source_max = tau - time_warping_para - 1
        source_min = tf.math.minimum(source_max - 1, time_warping_para)
        rand_source_time = tfv1.random_uniform(  # generate source points `t` of time axis between (W, tau-W)
            [], source_min, source_max, tf.int32)
        rand_dest_time = tfv1.random_uniform(  # generate dest points `t'` of time axis between (t-W, t+W), !!! if rand_dest_time == 0, might raise invertible error
            [], tf.math.maximum(tf.math.subtract(rand_source_time, time_warping_para), 1), tf.math.add(rand_source_time, time_warping_para), tf.int32)

        choosen_freq = choosen_freqs[i]
        sources.append([0, choosen_freq])
        sources.append([rand_source_time, choosen_freq])
        sources.append([tau, choosen_freq])

        dests.append([0, choosen_freq])
        dests.append([rand_dest_time, choosen_freq])
        dests.append([tau, choosen_freq])

    source_control_point_locations = tf.cast([sources], tf.float32)

    dest_control_point_locations = tf.cast([dests], tf.float32)

    # debug
    # spectrogram = tf.Print(spectrogram, [tf.shape(spectrogram)], message='spectrogram', first_n=1000)
    # spectrogram = tf.Print(spectrogram, sources, message='sources', first_n=1000)
    # spectrogram = tf.Print(spectrogram, dests, message='dests', first_n=1000)

    warped_spectrogram, _ = sparse_image_warp(spectrogram,
                                              source_control_point_locations=source_control_point_locations,
                                              dest_control_point_locations=dest_control_point_locations,
                                              interpolation_order=interpolation_order,
                                              regularization_weight=regularization_weight,
                                              num_boundary_points=num_boundary_points)
    return tf.reshape(warped_spectrogram, shape=(1, -1, freq_size))
