import re
from typing import AnyStr

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer

from ai_essay_prediction_misha.pipeline import wrappers
from ai_essay_prediction_misha.pipeline import entities_data

from nltk.tokenize import sent_tokenize

from collections import Counter


class LowerText(BaseEstimator, TransformerMixin):
    def __init__(self, text_column: str, column_out: str):
        self.text_column = text_column
        self.column_out = column_out

    def fit(self, x: pd.DataFrame, y=None):
        return self

    @wrappers.timed
    def transform(self, x: pd.DataFrame):
        x[self.column_out] = x[self.text_column].apply(lambda v: v.lower())
        return x


class CountAmountLettersInText(BaseEstimator, TransformerMixin):
    def __init__(self, text_column: str, column_out: str):
        self.text_column = text_column
        self.column_out = column_out

    def fit(self, x: pd.DataFrame, y=None):
        return self

    @wrappers.timed
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

    @wrappers.timed
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

    @wrappers.timed
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
    def __init__(self, columns_divided: list, column_divides: str, columns_out: list):
        print('sss')
        self.columns_divided = columns_divided
        self.column_divides = column_divides
        self.columns_out = columns_out

    def fit(self, x: pd.DataFrame, y=None):
        return self

    @wrappers.timed
    def transform(self, x: pd.DataFrame):
        self.columns_divided = set.union(*self.columns_divided)
        self.columns_out = set.union(*self.columns_out)

        x[list(self.columns_out)] = x[list(self.columns_divided)].div(x[self.column_divides], axis=0)
        return x


class TfidfVectorizerC(BaseEstimator, TransformerMixin):
    def __init__(self, vect, text_column: str, columns_out: set):
        self.vect = vect
        self.text_column = text_column
        self.columns_out = columns_out

    def fit(self, x: pd.DataFrame, y=None):
        self.vect.fit(x[self.text_column])
        feature_names_out = self.vect.get_feature_names_out()
        for value in feature_names_out:
            self.columns_out.add(value + '_tfidf')
        return self

    @wrappers.timed
    def transform(self, x: pd.DataFrame):
        data = self.vect.transform(x[self.text_column]).toarray()
        new_df = pd.DataFrame(columns=list(self.columns_out), data=data, index=x.index)
        x = pd.concat([x, new_df], axis=1)

        return x


class SeparatingTextIntoSentences(BaseEstimator, TransformerMixin):
    def __init__(self, text_column: str, column_out: str):
        self.text_column = text_column
        self.column_out = column_out

    def fit(self, x: pd.DataFrame, y=None):
        return self

    @wrappers.timed
    def transform(self, x: pd.DataFrame, y=None):
        text_vec = x[self.text_column]
        text_sentences_l = []

        for text in text_vec:
            text_sentences_l.append(sent_tokenize(text))

        x[self.column_out] = text_sentences_l
        return x


class CreatorEmotionsFromSentences(BaseEstimator, TransformerMixin):
    def __init__(self, sentences_column):
        self.classifier = entities_data.emotional_classifier
        self.sentences_column = sentences_column
        self.n_needed_sentences = 3

    def fit(self, x, y=None):
        return self

    @wrappers.timed
    def transform(self, x: pd.DataFrame, y=None):
        sentiment_df = pd.DataFrame(index=x.index)
        n_sentences_l = []

        for index, row in x.iterrows():
            sentences = row[self.sentences_column]
            random_indexes_l = []

            for i in range(100):
                random_idx = np.random.randint(0, len(sentences))
                if len(sentences[random_idx]) > 500:
                    continue
                if random_idx in random_indexes_l:
                    continue

                random_indexes_l.append(random_idx)
                if len(random_indexes_l) > self.n_needed_sentences: break;

            random_sentences_l = [sentences[i] for i in random_indexes_l]

            classified_l = self.classifier(random_sentences_l)

            for classified_value in classified_l:
                for label_score in classified_value:
                    label = label_score['label']
                    score = label_score['score']

                    if label not in sentiment_df.columns:
                        sentiment_df[label] = 0

                    sentiment_df.loc[index, label] += score

        return pd.concat([x, sentiment_df], axis=1)


class CreateAmountSentences(BaseEstimator, TransformerMixin):
    def __init__(self, column_with_sentences: str, column_out: str):
        self.column_with_sentences = column_with_sentences
        self.column_out = column_out

    def fit(self, x: pd.DataFrame, y=None):
        return self

    @wrappers.timed
    def transform(self, x: pd.DataFrame):
        n_sentences_ar = np.zeros(len(x))

        sentences_vec = x[self.column_with_sentences]

        for i, sentences in enumerate(sentences_vec):
            n_sentences_ar[i] += len(sentences)

        x[self.column_out] = n_sentences_ar
        return x


class DropperColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_drop: set):
        self.columns_to_drop = columns_to_drop

    def fit(self, x: pd.DataFrame, y=None):
        return self

    @wrappers.timed
    def transform(self, x: pd.DataFrame):
        return x.drop(columns=list(self.columns_to_drop))


class UserRegExp(BaseEstimator, TransformerMixin):
    def __init__(self, column_text: str, reg_exp: re.Pattern[AnyStr], columns_out: set, suffix: str):
        self.column_text = column_text
        self.reg_exp = reg_exp
        self.columns_out = columns_out
        self.suffix = suffix

    def fit(self, x: pd.DataFrame, y=None):
        return self

    @wrappers.timed
    def transform(self, x: pd.DataFrame):
        vec_of_vec = x[self.column_text].apply(lambda v: re.findall(self.reg_exp, v))
        result_dic = {}

        for i_element, element in enumerate(vec_of_vec):
            for key, value in dict(Counter(element)).items():
                result_key = key + self.suffix
                if key not in result_dic.keys():
                    result_dic[result_key] = np.zeros(len(vec_of_vec))
                    self.columns_out.add(result_key)

                result_dic[result_key][i_element] = value

        x = pd.concat([x, pd.DataFrame(result_dic)], axis=1)

        return x


class ChooserCorrelatedFeature(BaseEstimator, TransformerMixin):
    def __init__(self, columns: tuple, border: tuple = (-0.2, 0.2)):
        self.columns = columns
        self.border = border

    def fit(self, x: pd.DataFrame, y=None):
        return self

    @wrappers.timed
    def transform(self, x: pd.DataFrame):
        cor_vec = x[self.columns].corrwith(y)

        borders = (-0.2, 0.2)
        needed_columns = cor_vec[(cor_vec < borders[0]) | (cor_vec > borders[1])].index
        needed_columns

        return x
