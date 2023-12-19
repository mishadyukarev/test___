from sklearn.feature_extraction.text import TfidfVectorizer

################################################
CREATE_FOR_SUBMISSION = False  # Change it only
################################################

# language_tool = language_tool_python.LanguageTool('en-US')

# freq_corr_df = None
# name_of_emotional_model = 'emotional_classifier'

# NAME_OF_TFIDF_VECT = 'tfidf_vectorizer.pkl'

# if create_for_submission:
#    pass
# NOTEBOOK_REF = '/kaggle/input/feature-engineering-by-ai_essay_prediction_misha/'

# model_ref = NOTEBOOK_REF + name_of_emotional_model
# emotional_classifier = pipeline("text-classification", model=model_ref, return_all_scores=True)

# freq_corr_df = pd.read_csv(NOTEBOOK_REF + 'ruleId_frequency_correlation.csv')
# with open(NOTEBOOK_REF + NAME_OF_TFIDF_VECT, 'rb') as f:
#    tfidf_vectorizer = load(f)

# else:
# pass
# name_of_emotional_model = 'emotional_classifier'
# classifier = pipeline('sentiment-analysis', model='arpanghoshal/EmoRoBERTa') # first model
# emotional_classifier = pipeline("text-classification",
#                                model='j-hartmann/emotion-english-distilroberta-base',
#                                return_all_scores=True)
# emotional_classifier.save_pretrained('/kaggle/working/' + name_of_emotional_model)

# freq_corr_df = pd.read_csv(
#    '/kaggle/input/get-ruleid-frequency-correlation/ruleId_frequency_correlation.csv')
# freq_corr_df.to_csv('ruleId_frequency_correlation.csv', index=False)

# tfidf_vectorizer = TfidfVectorizer(ngram_range=(2, 5), max_features=1000,
#                                   use_idf=False)  ##sublinear_tf=True#,, ,  stop_words="english" stop_words='english', max_features=6000

# freq_border = 0.05
# corr_border = 0.05
# needed_fc_df = freq_corr_df[(freq_corr_df['frequency'] > freq_border) & (
#            (freq_corr_df['correlation'] > corr_border) | (
#                freq_corr_df['correlation'] < -corr_border))]  # | df['correlation']<-0.1
# print(f'Amount rudeId feature we take: {len(needed_fc_df)}')
