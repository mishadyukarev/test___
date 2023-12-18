import misha.pipeline.classes as classes
import misha.pipeline.names as n

from sklearn.pipeline import Pipeline

def create_main_pipeline(create_for_submission):
    # columns_tfidf_out = []
    n_every_letter_columns_set = set()

    pipeline_l = [
        # ('add_has_mistake_columns', AddHasMistakesColumns(language_tool, 'text', True, needed_fc_df.ruleId.tolist())),
        # ('correcting_sentences', CorrectorSentences('text', 'corrected_text')),

        # ('separating_text_into_sentences', SeparatingTextIntoSentences('corrected_text', 'sentences_corrected_text')),
        # ('create_amount_words', CreatorAmountWords('corrected_text', 'amount_words')),
        # ('create_extra_features', CreatorAdditionRegressionFeatures('corrected_text', 'amount_words', [])),
        # ('create_emotions', CreatorEmotionsFromSentences(emotional_classifier, 'sentences_corrected_text')),

        ('lower_text', classes.LowerText(n.Columns.TEXT, n.Columns.TEXT)),
        ('count_amount_letters', classes.CountAmountLettersInText(n.Columns.TEXT, 'n_letters')),
        ('count_amount_every_letter',
         classes.CountAmountEveryLetterInText(n.Columns.TEXT, n_every_letter_columns_set)),
        ('divide_matrix_into_vector',
         classes.DivideMatrixIntoVector(n_every_letter_columns_set, 'n_letters', n_every_letter_columns_set)),
        ('remove_less_popular_features', classes.RemoveLessPopularFeatures(n_every_letter_columns_set, 0.4)),

        ('tfidf_vectorizer', classes.TfidfVectorizerC(n.Columns.TEXT, set())),
    ]

    return Pipeline(pipeline_l)
