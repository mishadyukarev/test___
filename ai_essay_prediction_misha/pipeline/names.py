
from collections import namedtuple
from dataclasses import dataclass

#Columns = namedtuple \
#    ('Constants', ['TEXT', 'CORRECTED_TEXT', 'SENTENCES_CORRECTED_TEXT']) \
#    ('text', 'corrected_text', 'sentences_corrected_text')


@dataclass(frozen=True)
class _Columns:
    TEXT: str = 'text'
    CORRECTED_TEXT: str = 'corrected_text'
    SENTENCES_CORRECTED_TEXT: str = 'sentences_corrected_text'


Columns = _Columns()

