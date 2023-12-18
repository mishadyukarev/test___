import misha.pipeline.classes as classes

from sklearn.pipeline import Pipeline

#columns_tfidf_out = []
n_every_letter_columns_l = []

TEXT_NAME = 'text'

pipeline_l = [
    # ('add_has_mistake_columns', AddHasMistakesColumns(language_tool, 'text', True, needed_fc_df.ruleId.tolist())),
    #('correcting_sentences', CorrectorSentences('text', 'corrected_text')),

    # ('separating_text_into_sentences', SeparatingTextIntoSentences('corrected_text', 'sentences_corrected_text')),
    # ('create_amount_words', CreatorAmountWords('corrected_text', 'amount_words')),
    # ('create_extra_features', CreatorAdditionRegressionFeatures('corrected_text', 'amount_words', [])),
    # ('create_emotions', CreatorEmotionsFromSentences(emotional_classifier, 'sentences_corrected_text')),

    ('lower_text', classes.LowerText(TEXT_NAME, TEXT_NAME)),
    ('count_amount_letters', classes.CountAmountLettersInText(TEXT_NAME, 'n_letters')),
    ('count_amount_every_letter',
     classes.CountAmountEveryLetterInText(TEXT_NAME, n_every_letter_columns_l)),
    ('divide_matrix_into_vector',
     classes.DivideMatrixIntoVector(n_every_letter_columns_l, 'n_letters', n_every_letter_columns_l)),
    ('remove_less_popular_features', classes.RemoveLessPopularFeatures(n_every_letter_columns_l, 0.4)),

    # ('tfidf_vectorizer', TfidfVectorizerC('corrected_text', columns_tfidf_out)),
]

pipeline = Pipeline(pipeline_l)
