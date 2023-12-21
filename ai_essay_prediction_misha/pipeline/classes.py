import re
from typing import AnyStr

import numpy as np
import pandas as pd
from IPython.core.display import display


from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer

from ai_essay_prediction_misha.pipeline import wrappers, entities_data
from ai_essay_prediction_misha.pipeline.abstract_classes import TextColumnOutWorker, TextColumnsOutWorker

from nltk.tokenize import sent_tokenize

from collections import Counter


class LowerText(TextColumnOutWorker, BaseEstimator, TransformerMixin):
    def __init__(self, _column_text: str, _column_out: str):
        super().__init__(_column_text, _column_out)

    def fit(self, x: pd.DataFrame, y=None):
        return self

    @wrappers.timed
    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        x[self._column_out] = x[self._column_text].apply(lambda v: v.lower())
        return x


class CountAmountLettersInText(TextColumnOutWorker, BaseEstimator, TransformerMixin):
    def __init__(self, _column_text: str, _column_out: str):
        super().__init__(_column_text, _column_out)

    def fit(self, x: pd.DataFrame, y=None):
        return self

    @wrappers.timed
    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        x[self._column_out] = x[self._column_text].apply(lambda v: len(v))
        return x


class CountAmountEveryLetterInText(TextColumnsOutWorker, BaseEstimator, TransformerMixin):
    def __init__(self, _column_text: str, _columns_out: set):
        super().__init__(_column_text, _columns_out)

    def fit(self, x: pd.DataFrame, y=None):
        for text in x[self._column_text]:
            for letter in text:
                if letter not in self._columns_out:
                    self._columns_out.add(letter)
        return self

    @wrappers.timed
    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        n_every_letter_dic = {key: np.zeros(len(x)) for key in self._columns_out}

        for i, text in enumerate(x[self._column_text]):
            for letter in text:
                if letter in self._columns_out:
                    n_every_letter_dic[letter][i] += 1

        x = pd.concat([x, pd.DataFrame(n_every_letter_dic, index=x.index)], axis=1)

        return x


class RemoveLessPopularFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, __columns: set, __border: float, __columns_stayed: set):
        self.__columns = __columns
        self.__border = __border
        self.__columns_stayed = __columns_stayed

    def fit(self, x: pd.DataFrame, y=None):
        self.__columns = tuple(self.__columns)
        return self

    @wrappers.timed
    def transform(self, x: pd.DataFrame):
        frequency_dic = {}

        for column in self.__columns:
            frequency_dic[column] = sum(x[column].apply(lambda v: int(v > 0))) / len(x)

        n_letters_vec = pd.Series(index=frequency_dic.keys(), data=frequency_dic.values())
        not_needed_columns = set(n_letters_vec[n_letters_vec < self.__border].index)
        print(f'not needed columns: {not_needed_columns}')
        x.drop(columns=list(not_needed_columns), inplace=True)

        for value in set(self.__columns) - not_needed_columns:
            self.__columns_stayed.add(value)
        return x


class DivideMatrixIntoVector(BaseEstimator, TransformerMixin):
    def __init__(self, __columns_divided: set[str], __column_divides: str, __columns_out: set[str]):
        self.__columns_divided = __columns_divided
        self.__column_divides = __column_divides
        self.__columns_out = __columns_out

    def fit(self, x: pd.DataFrame, y=None):
        self.__columns_divided = tuple(self.__columns_divided)
        self.__columns_out = tuple(self.__columns_out)
        return self

    @wrappers.timed
    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        x[np.array(self.__columns_out)] = x[np.array(self.__columns_divided)].div(x[self.__column_divides], axis=0)
        return x


class DropperColumns(BaseEstimator, TransformerMixin):
    def __init__(self, __columns_to_drop: set[str]):
        self.__columns_to_drop = __columns_to_drop

    def fit(self, x: pd.DataFrame, y=None):
        return self

    @wrappers.timed
    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        return x.drop(columns=list(self.__columns_to_drop))



class TfidfVectorizerC(TextColumnsOutWorker, BaseEstimator, TransformerMixin):
    def __init__(self, vect, _column_text: str, _columns_out: set):
        super().__init__(_column_text, _columns_out)
        self.__vect = vect

    def fit(self, x: pd.DataFrame, y=None):
        self.__vect.fit(x[self._column_text])
        feature_names_out = self.__vect.get_feature_names_out()
        for value in feature_names_out:
            self._columns_out.add(value + '_tfidf')
        return self

    @wrappers.timed
    def transform(self, x: pd.DataFrame):
        data = self.__vect.transform(x[self._column_text]).toarray()
        new_df = pd.DataFrame(columns=list(self._columns_out), data=data, index=x.index)
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


class CreatorColumnsByUsingManyRegularExpressions(TextColumnsOutWorker, BaseEstimator, TransformerMixin):
    def __init__(self, _column_text: str, __patterns_reg_exp: set[re.Pattern[AnyStr]], _columns_out: set):
        super().__init__(_column_text=_column_text, _columns_out=_columns_out)
        self.__patterns_reg_exp = __patterns_reg_exp

    def fit(self, x: pd.DataFrame, y=None):
        return self

    @wrappers.timed
    def transform(self, x: pd.DataFrame):
        for i_pattern, pattern in enumerate(self.__patterns_reg_exp):
            name_new_column = f'pattern_{i_pattern}_{pattern.pattern}'
            self._columns_out.add(name_new_column)
            x[name_new_column] = x[self._column_text].apply(lambda v: pattern.findall(v))
        return x


class Create(TextColumnsOutWorker, BaseEstimator, TransformerMixin):
    def __init__(self, __columns, _column_text: str, _columns_out: set):
        super().__init__(_column_text, _columns_out)
        self.__columns = __columns

    def fit(self, x: pd.DataFrame, y=None):
        return self

    @wrappers.timed
    def transform(self, x: pd.DataFrame):
        result_dic = {}
        for i_column, vec_column in enumerate(self.__columns):
            vec_of_vec = x[vec_column]

            for i_row, row_value_l in enumerate(vec_of_vec):
                for key, value in dict(Counter(row_value_l)).items():
                    result_key = key + f'_{i_column}'
                    if result_key not in result_dic.keys():
                        result_dic[result_key] = np.zeros(len(x))
                        self._columns_out.add(result_key)
                    result_dic[result_key][i_row] = value

        x = pd.concat([x, pd.DataFrame(result_dic, index=x.index)], axis=1)

        return x


'''        for reg_exp_pattern in self.patterns_reg_exp:
            vec_of_vec = x[self.column_text].apply(lambda v: re.findall(reg_exp_pattern, v))
            result_dic = {}

            for i_element, element in enumerate(vec_of_vec):
                for key, value in dict(Counter(element)).items():
                    result_key = key + self.suffix
                    if key not in result_dic.keys():
                        result_dic[result_key] = np.zeros(len(vec_of_vec))
                        self.columns_out.add(result_key)

                    result_dic[result_key][i_element] = value

            x = pd.concat([x, pd.DataFrame(result_dic)], axis=1)'''

'''class ChooserCorrelatedFeature(BaseEstimator, TransformerMixin):
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

        return x''';
