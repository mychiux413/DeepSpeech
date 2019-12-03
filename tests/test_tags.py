
# -*- coding: utf-8 -*-
from util.tags import Tags
import os

TAGS_FILENAME = 'data/tags.txt'
TMP_TAGS_FILENAME = 'data/tmp_tags.txt'


def test_tags():
    tags = Tags(TAGS_FILENAME)

    assert tags.max_label_size() == 512
    assert tags.encode('not exists')[0] == 1.0

    current_size = tags.size()
    tags.add_tag(' new tag ')
    assert tags.size() == current_size + 1

    # **** savable and reloadable ****
    encoded = tags.encode('new_tag')
    assert encoded[tags.size()-1] == 0.5
    assert len(encoded) == tags.max_label_size()

    tags.save(TMP_TAGS_FILENAME)
    reload_tags = Tags(TMP_TAGS_FILENAME)
    assert reload_tags.encode(['new-tag'])[tags.size()-1] == 0.5
    os.remove(TMP_TAGS_FILENAME)
    # ********************************

    try:
        for i in range(tags.max_label_size()):
            tags.add_tag(str(i))
        raise AssertionError('must throw error when tags out of limit')
    except IndexError:
        pass
