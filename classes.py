import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin


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
    def __init__(self, text_column, column_out):
        self.text_column = text_column
        self.column_out = column_out

    def fit(self, x: pd.DataFrame, y=None):
        return self

    def transform(self, x: pd.DataFrame):
        x[self.column_out] = x[self.text_column].apply(lambda v: len(v))
        return x


class CountAmountEveryLetterInText(BaseEstimator, TransformerMixin):
    def __init__(self, text_column, columns_out):
        self.text_column = text_column
        self.columns_out = columns_out
        self.needed_letters = self.columns_out

    def fit(self, x: pd.DataFrame, y=None):
        for i, text in enumerate(x[self.text_column]):
            for letter in text:
                if letter not in self.needed_letters:
                    self.needed_letters.append(letter)

        return self

    def transform(self, x: pd.DataFrame):
        n_every_letter_dic = {}

        for i, text in enumerate(x[self.text_column]):
            for letter in text:
                if letter in self.needed_letters:
                    if letter in n_every_letter_dic.keys():
                        n_every_letter_dic[letter][i] += 1
                    else:
                        n_every_letter_dic[letter] = [0] * len(x)

        x = pd.concat([x, pd.DataFrame(n_every_letter_dic)], axis=1)

        return x


class RemoveLessPopularFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, columns, border):
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
        print(not_needed_columns)
        x.drop(columns=not_needed_columns, inplace=True)

        return x


class DivideMatrixIntoVector(BaseEstimator, TransformerMixin):
    def __init__(self, columns_divided, column_divides, columns_out):
        self.columns_divided = columns_divided
        self.column_divides = column_divides
        self.columns_out = columns_out

    def fit(self, x: pd.DataFrame, y=None):
        return self

    def transform(self, x: pd.DataFrame):
        x[self.columns_out] = x[self.columns_divided].div(x[self.column_divides], axis=0)
        return x
