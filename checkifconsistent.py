import joblib
from data_processing import remove_custom_stopwords

tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')
vocabulary = joblib.load('tfidf_vocabulary.pkl')

tfidf_vectorizer.vocabulary_ = vocabulary

new_file_path = '/Users/fatimamuhammed/Downloads/archive/bbc/sport/016.txt'

with open(new_file_path, 'r', encoding='utf-8') as file:
    new_text = file.read()

new_text = remove_custom_stopwords(new_text)

new_text_tfidf = tfidf_vectorizer.transform([new_text])

if set(tfidf_vectorizer.get_feature_names_out()) == set(vocabulary.keys()):
    print("Vocabulary is consistent.")
else:
    print("Vocabulary mismatch detected.")
