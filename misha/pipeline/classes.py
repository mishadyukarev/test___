import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer


class LowerText(BaseEstimator, TransformerMixin):
    def __init__(self, text_column: str, column_out: str):
        self.text_column = text_column
        self.column_out = column_out

    def fit(self, x: pd.DataFrame, y=None):
        return self

    def transform(self, x: pd.DataFrame):
        x[self.column_out] = x[self.text_column].apply(lambda v: v.lower())
        return x


class CountAmountLettersInText(BaseEstimator, TransformerMixin):
    def __init__(self, text_column: str, column_out: str):
        self.text_column = text_column
        self.column_out = column_out

    def fit(self, x: pd.DataFrame, y=None):
        return self

    def transform(self, x: pd.DataFrame):
        x[self.column_out] = x[self.text_column].apply(lambda v: len(v))
        return x


class CountAmountEveryLetterInText(BaseEstimator, TransformerMixin):
    def __init__(self, text_column: str, columns_out: set):
        self.text_column = text_column
        self.columns_out = columns_out

    def fit(self, x: pd.DataFrame, y=None):
        for i, text in enumerate(x[self.text_column]):
            for letter in text:
                if letter not in self.columns_out:
                    self.columns_out.add(letter)

        return self

    def transform(self, x: pd.DataFrame):
        n_every_letter_dic = {key: [0] * len(x) for key in self.columns_out}

        for i, text in enumerate(x[self.text_column]):
            for letter in text:
                if letter in self.columns_out:
                    n_every_letter_dic[letter][i] += 1

        x = pd.concat([x, pd.DataFrame(n_every_letter_dic)], axis=1)

        return x


class RemoveLessPopularFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, columns: set, border: float):
        self.columns = columns
        self.border = border

    def fit(self, x: pd.DataFrame, y=None):
        return self

    def transform(self, x: pd.DataFrame):
        frequency_dic = {}

        for column in self.columns:
            frequency_dic[column] = sum(x[column].apply(lambda v: int(v > 0))) / len(x)

        n_letters_vec = pd.Series(index=frequency_dic.keys(), data=frequency_dic.values())
        not_needed_columns = n_letters_vec[n_letters_vec < self.border].index
        print(f'not needed columns: {list(not_needed_columns)}')
        x.drop(columns=not_needed_columns, inplace=True)

        return x


class DivideMatrixIntoVector(BaseEstimator, TransformerMixin):
    def __init__(self, columns_divided: set, column_divides: str, columns_out: set):
        self.columns_divided = list(columns_divided)
        self.column_divides = column_divides
        self.columns_out = list(columns_out)

    def fit(self, x: pd.DataFrame, y=None):
        return self

    def transform(self, x: pd.DataFrame):
        x[self.columns_out] = x[self.columns_divided].div(x[self.column_divides], axis=0)
        return x


class TfidfVectorizerC(BaseEstimator, TransformerMixin):
    def __init__(self, text_column: str, columns_out: set):
        self.vect = TfidfVectorizer(ngram_range=(2, 5), max_features=1000, use_idf=False)
        self.text_column = text_column
        self.columns_out = columns_out

    def fit(self, x: pd.DataFrame, y=None):
        self.vect.fit(x[self.text_column])
        feature_names_out = self.vect.get_feature_names_out()
        for value in feature_names_out:
            self.columns_out.add(value + '_tfidf')
        return self

    def transform(self, x: pd.DataFrame):
        data = self.vect.transform(x[self.text_column]).toarray()
        new_df = pd.DataFrame(columns=list(self.columns_out), data=data, index=x.index)
        x = pd.concat([x, new_df], axis=1)

        return x
