from __future__ import absolute_import, division, print_function
import tensorflow as tf
import tensorflow.compat.v1 as tfv1
import numpy as np
from tempfile import NamedTemporaryFile
import os
import shutil
from util.logging import log_info
from util.flags import FLAGS

def to_norm_lengths(lengths, alpha, beta, dynamic=True):
    if dynamic:
        return tf.pow((beta + tf.cast(lengths, dtype=tf.float32) / (beta + 1)), alpha)

    if isinstance(lengths, list):
        lengths = np.array(lengths, dtype=float)
    elif isinstance(lengths, int):
        lengths = float(lengths)
    return np.power((beta + lengths) / (beta + 1), alpha)


def tune(train, test):
    log_info("Enable Length Normalization Experiment")
    assert FLAGS.test_files
    assert FLAGS.train_files
    assert FLAGS.len_norm_avg_logits_length > 0
    assert FLAGS.len_norm_avg_transcript_length > 0
    assert FLAGS.early_stop == False

    results = {
        'raw_data': [],
        'best_result': {},
        'parameters': {
            'len_norm_avg_logits_length': FLAGS.len_norm_avg_logits_length,
            'len_norm_avg_transcript_length': FLAGS.len_norm_avg_transcript_length,
            'learning_rate': FLAGS.learning_rate,
            'len_norm_exp_logits_alpha_radius': FLAGS.len_norm_exp_logits_alpha_radius,
            'len_norm_exp_logits_beta_radius': FLAGS.len_norm_exp_logits_beta_radius,
            'len_norm_exp_transcript_alpha_radius': FLAGS.len_norm_exp_transcript_alpha_radius,
            'len_norm_exp_transcript_beta_radius': FLAGS.len_norm_exp_transcript_beta_radius,
            'len_norm_exp_iterations': FLAGS.len_norm_exp_iterations,
        }
    }

    if FLAGS.load != "init":
        backup = CheckpointBackup(FLAGS.checkpoint_dir)

    logits_alpha = best_logits_alpha = FLAGS.len_norm_logits_alpha
    logits_beta = best_logits_beta = FLAGS.len_norm_logits_beta
    transcript_alpha = best_transcript_alpha = FLAGS.len_norm_transcript_alpha
    transcript_beta = best_transcript_beta = FLAGS.len_norm_transcript_beta

    # random search
    best_batch_wer = 1.0
    for i in range(FLAGS.len_norm_exp_iterations):
        if i > 0:
            coords = np.random.normal(size=4)
            length = np.math.sqrt(sum(coords**2))
            rand_points = coords / length

            logits_alpha = best_logits_alpha + rand_points[0] * FLAGS.len_norm_exp_logits_alpha_radius
            logits_beta = best_logits_beta + rand_points[1] * FLAGS.len_norm_exp_logits_beta_radius
            transcript_alpha = best_transcript_alpha + rand_points[2] * FLAGS.len_norm_exp_transcript_alpha_radius
            transcript_beta = best_transcript_beta + rand_points[3] * FLAGS.len_norm_exp_transcript_beta_radius

            # constraint params in boundary
            logits_alpha = max(min(logits_alpha, 1.0), 0.0)
            logits_beta = max(logits_beta, 0.0)
            transcript_alpha = max(min(transcript_alpha, 1.0), 0.0)
            transcript_beta = max(transcript_beta, 0.0)

            log_info("set logits alpha/beta, transcript alpha/beta as:\n{}/{} {}/{}".format(
                logits_alpha, logits_beta, transcript_alpha, transcript_beta
            ))
            FLAGS.len_norm_logits_alpha = logits_alpha
            FLAGS.len_norm_logits_beta = logits_beta
            FLAGS.len_norm_transcript_alpha = transcript_alpha
            FLAGS.len_norm_transcript_beta = transcript_beta

        log_info("reset random seed as: {}".format(FLAGS.random_seed))
        tfv1.set_random_seed(FLAGS.random_seed)
        sorts = FLAGS.train_files_sortby.split(',')
        for j, sortby in enumerate(sorts):
            tfv1.reset_default_graph()
            if sortby.find(':') > -1:
                sortby, epoch = sortby.split(':')
                epoch = int(epoch)
            else:
                epoch = FLAGS.epochs
            load = FLAGS.load if j == 0 else 'last'
            train(epoch, load, sortby)

        tfv1.reset_default_graph()
        samples = test()
        word_distances = [sample['word_distance'] for sample in samples]
        word_lengths = [sample['word_length'] for sample in samples]
        batch_wer = np.sum(word_distances) / np.sum(word_lengths)
        mean_loss = np.mean([sample['loss'] for sample in samples])

        if batch_wer < best_batch_wer:
            log_info("found better wer: {} < {}".format(batch_wer, best_batch_wer))
            best_batch_wer = batch_wer
            best_logits_alpha = logits_alpha
            best_logits_beta = logits_beta
            best_transcript_alpha = transcript_alpha
            best_transcript_beta = transcript_beta

        results['raw_data'].append({
            'logits_alpha': float(logits_alpha),
            'logits_beta': float(logits_beta),
            'transcript_alpha': float(transcript_alpha),
            'transcript_beta': float(transcript_beta),
            'batch_wer': float(batch_wer),
            'mean_loss': float(mean_loss),
        })

        if FLAGS.load != "init":
            backup.restore()

    log_info("Final Parameters")
    log_info("Logits Alpha: {}".format(best_logits_alpha))
    log_info("Logits Beta: {}".format(best_logits_beta))
    log_info("Transcript Alpha: {}".format(best_transcript_alpha))
    log_info("Transcript Beta: {}".format(best_transcript_beta))

    results['best_result'] = {
        'Logits Alpha': best_logits_alpha,
        'Logits Beta': best_logits_beta,
        'Transcript Alpha': best_transcript_alpha,
        'Transcript Beta': best_transcript_beta,
        'WER': best_batch_wer,
        'Use Batch Sequence': FLAGS.len_norm_use_batch_sequence,
    }

    return results


class CheckpointBackup:

    def __init__(self, checkpoint_dir):
        self.checkpoint_dir = checkpoint_dir
        self._backups = {}
        self.backup_checkpoint()

    def backup_checkpoint(self):
        for dirpath, _, filenames in os.walk(self.checkpoint_dir):
            for filename in filenames:
                write_filemode = 'w+b'
                read_filemode = 'rb'
                if filename.endswith(('checkpoint', '.txt')):
                    write_filemode = 'w+'
                    read_filemode = 'r'
                filepath = os.path.abspath(os.path.join(dirpath, filename))

                log_info('Backup checkpoint file: {}'.format(filepath))
                tmp_file = NamedTemporaryFile(mode=write_filemode)
                with open(filepath, read_filemode) as f:
                    shutil.copyfileobj(f, tmp_file.file)

                tmp_file.file.seek(0, 0)
                self._backups[filepath] = tmp_file

    def restore(self):
        log_info('restore checkpoint files')
        for dstpath, tmp_file in self._backups.items():
            write_filemode = 'wb'
            if dstpath.endswith(('checkpoint', '.txt')):
                write_filemode = 'w'

            with open(dstpath, write_filemode) as dst_f:
                shutil.copyfileobj(tmp_file.file, dst_f)
            tmp_file.file.seek(0, 0)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.restore()

        log_info("Exit Checkpoint Backup")
        for _, tmp_file in self._backups.items():
            tmp_file.close()
