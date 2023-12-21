from collections import namedtuple

Columns = namedtuple \
    ('Constants', ['TEXT', 'CORRECTED_TEXT', 'SENTENCES_CORRECTED_TEXT']) \
    ('text', 'corrected_text', 'sentences_corrected_text')
