import re

import ai_essay_prediction_misha.pipeline.classes as classes
import ai_essay_prediction_misha.pipeline.names as n

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.pipeline import Pipeline

from transformers import pipeline



def create_main_pipeline(create_for_submission):
    # columns_tfidf_out = []
    n_every_letter_columns_set = set()

    # vectorizer = TfidfVectorizer(ngram_range=(4, 6), max_features=10000, use_idf=False)  # , use_idf=False

    reg_expressions_l = {
        re.compile(r"[a-z]+([a-z]{2})\W"),
        re.compile(r"[a-z]+([a-z]{3})\W"),
        re.compile(r"([a-z]\W[a-z])"),
    }

    classes.UserManyRegExp('', reg_expressions_l, set())

    pipeline_l = [
        # ('add_has_mistake_columns', AddHasMistakesColumns(language_tool, 'text', True, needed_fc_df.ruleId.tolist())),
        # ('correcting_sentences', CorrectorSentences('text', 'corrected_text')),

        # ('separating_text_into_sentences', SeparatingTextIntoSentences('corrected_text', 'sentences_corrected_text')),
        # ('create_amount_words', CreatorAmountWords('corrected_text', 'amount_words')),
        # ('create_extra_features', CreatorAdditionRegressionFeatures('corrected_text', 'amount_words', [])),
        # ('create_emotions', CreatorEmotionsFromSentences(emotional_classifier, 'sentences_corrected_text')),

        ('lower_text',
         classes.LowerText(n.Columns.TEXT,
                           n.Columns.CORRECTED_TEXT)),

        ('count_amount_letters',
         classes.CountAmountLettersInText(n.Columns.CORRECTED_TEXT,
                                          'n_letters')),

        ('count_amount_every_letter',
         classes.CountAmountEveryLetterInText(n.Columns.CORRECTED_TEXT,
                                              n_every_letter_columns_set)),

        # ('remove_less_popular_features',
        # classes.RemoveLessPopularFeatures(n_every_letter_columns_set, 0.4)),

        # ('tfidf_vectorizer', classes.TfidfVectorizerC(vectorizer, n.Columns.CORRECTED_TEXT, set())),

        ('separating_text_into_sentences',
         classes.SeparatingTextIntoSentences(n.Columns.CORRECTED_TEXT,
                                             n.Columns.SENTENCES_CORRECTED_TEXT)),

        # ('create_emotions', classes.CreatorEmotionsFromSentences(n.Columns.SENTENCES_CORRECTED_TEXT)),
        ('create_n_words',
         classes.CreateAmountSentences(n.Columns.SENTENCES_CORRECTED_TEXT,
                                       'n_sentences')),

        # ('s1',
        # classes.UserRegExp(n.Columns.CORRECTED_TEXT,
        #                   re.compile(r"[a-zA-Z]+([a-z])\W"),
        #                   cols_reg_exp_1_set)),

    ]

    pipeline_l += [
        ('reg_expr', classes.UserManyRegExp(n.Columns.CORRECTED_TEXT, reg_expressions_l, ))
    ]



    return Pipeline(pipeline_l)


def create_cleaning_outliers_pipeline():
    pipeline_l = [

    ]

    return Pipeline(pipeline_l)


'''        ('s2',
         classes.UserRegExp(n.Columns.CORRECTED_TEXT,
                            re.compile(r"[a-z]+([a-z]{2})\W"),
                            cols_reg_exp_2_set, '_0')),

        ('s3',
         classes.UserRegExp(n.Columns.CORRECTED_TEXT,
                            re.compile(r"[a-z]+([a-z]{3})\W"),
                            cols_reg_exp_3_set, '_1')),

        ('4',
         classes.UserRegExp(n.Columns.CORRECTED_TEXT,
                            re.compile(r"([a-z]\W[a-z])"),
                            cols_reg_exp_4_set, '_2')),

        ('finding_word_with_one_letter',
         classes.UserRegExp(n.Columns.CORRECTED_TEXT,
                            re.compile(r"\W(\w)\W"),
                            cols_reg_exp_5_set, '_3')),

        ('5',
         classes.UserRegExp(n.Columns.CORRECTED_TEXT,
                            re.compile(r"\W(\w{2})\W"),
                            cols_reg_exp_6_set, '_4')),

        ('6',
         classes.UserRegExp(n.Columns.CORRECTED_TEXT,
                            re.compile(r"['\"]\w\s"),
                            cols_reg_exp_7_set, '_6')),'''

'''pipeline_l += [
    ('remove_less_popular_features',
     classes.RemoveLessPopularFeatures(set.union(*reg_exp_out_l), 0.4)),
]

pipeline_l += [('divide_matrix_into_vector',
                classes.DivideMatrixIntoVector([n_every_letter_columns_set,
                                                set.union(*reg_exp_out_l),
                                                ],
                                               'n_letters',
                                               [n_every_letter_columns_set,
                                                set.union(*reg_exp_out_l),
                                                ])),

               ('drop',
                classes.DropperColumns({n.Columns.SENTENCES_CORRECTED_TEXT, 'n_sentences', 'n_letters'})),
               ]'''
