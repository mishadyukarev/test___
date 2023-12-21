from sklearn.base import BaseEstimator, TransformerMixin
from abc import ABC, abstractmethod


class TextColumnsOutWorker(ABC, BaseEstimator, TransformerMixin):
    def __init__(self, _column_text: str, _columns_out: set):
        self._column_text = _column_text
        self._columns_out = _columns_out


class TextColumnOutWorker(ABC, BaseEstimator, TransformerMixin):
    def __init__(self, _column_text: str, _column_out: str):
        self._column_text = _column_text
        self._column_out = _column_out
