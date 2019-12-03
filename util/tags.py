from __future__ import absolute_import, division, print_function

import codecs
import numpy as np
import re
import os

INVALID_CHAR_PATTERN = re.compile('[^a-z0-9]', re.IGNORECASE)


def normalize_tag(string):
    return INVALID_CHAR_PATTERN.sub(' ', string.strip().lower())


class Tags(object):
    def __init__(self, config_file):
        self._config_file = config_file
        self._label_to_str = {}
        self._str_to_label = {}
        self._size = 1  # 0 is default tag
        self._max_label_size = None
        self._zeros = None
        if config_file:
            with codecs.open(config_file, 'r', 'utf-8') as fin:
                for i, line in enumerate(fin):
                    if i == 0:
                        try:
                            self.set_max_lable_size(
                                int(normalize_tag(line).strip()))
                        except ValueError:
                            print('set max label size as default value: 512')
                            self.set_max_lable_size(512)
                        continue
                    if line[0:2] == '\\#':
                        line = '#\n'
                    elif line[0] == '#':
                        continue
                    self.add_tag(line)

    def _string_from_label(self, label):
        return self._label_to_str[label]

    def _label_from_string(self, string):
        """if string doesn't exist, label is 0"""
        return self._str_to_label.get(normalize_tag(string), 0)

    def add_tag(self, string):
        string = normalize_tag(string)
        assert string not in self._str_to_label, 'tag: {} has already exist'.format(
            string)
        if self._size == self._max_label_size:
            raise IndexError('tags size is full: {}'.format(self._size))

        self._label_to_str[self._size] = string
        self._str_to_label[string] = self._size
        self._size += 1

    def encode(self, tags):
        """
        Returns:
            vector: encoded numpy array with shape [`max_label_size`]
        """
        if isinstance(tags, str):
            tags = tags.split(',')
        assert isinstance(tags, list), 'got data type: {}'.format(type(tags))
        labels = list(set([self._label_from_string(tag)
                           for tag in tags] + [0]))
        fill_value = 1.0 / len(labels)
        vector = self._zeros.copy()
        for index in labels:
            vector[index] = fill_value
        return vector

    def decode(self, labels):
        if isinstance(labels, int):
            labels = [labels]
        assert isinstance(
            labels, list), 'got data type: {}'.format(type(labels))
        return [self._string_from_label(label) for label in list(set(labels))]

    def serialize(self):
        raise NotImplementedError(
            "Tags Class has no serialize method, maybe you want to use Alphabet Class?")

    def size(self):
        return self._size

    def config_file(self):
        return self._config_file

    def max_label_size(self):
        return self._max_label_size

    def reset(self):
        return Tags(self.config_file())

    def save(self, filename):
        if os.path.exists(filename):
            print(
                '[OVERWRITE WARNING] the tags file has already exists: {}'.format(filename))
        with codecs.open(filename, 'w+', 'utf-8') as fout:
            fout.write('# {}\n'.format(self._max_label_size))
            fout.write(
                '# this is generated tags file, head number is max tags size\n')
            for i in range(self._size):
                if i == 0:  # preserved label
                    continue
                fout.write('{}\n'.format(self._string_from_label(i)))
            fout.write(
                '# The last (non-comment) line needs to end with a newline.\n')

    def set_max_lable_size(self, size):
        assert self._size < size, 'the max size must greater than current size: {}, but got {}'.format(
            self._size, size)
        self._max_label_size = size
        self._zeros = np.zeros((self._max_label_size,), dtype=np.float32)
        if self._size > 1:
            print('[WARNING] set max label size as: {}, this might cause your model crashed'.format(
                self._max_label_size))
