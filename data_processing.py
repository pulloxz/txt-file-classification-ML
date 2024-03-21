from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import os


#dataset https://www.kaggle.com/datasets/jensenbaxter/10dataset-text-document-classification/
dataset_directories = ['dataset/sport', 'dataset/business', 'dataset/entertainment', 'dataset/food',
                       'dataset/technology', 'dataset/space', 'dataset/politics', 'dataset/medical',
                       'dataset/historical', 'dataset/graphics']

all_document_texts = []
custom_stopwords_file = 'stopword'

with open(custom_stopwords_file, 'r') as stopword_file:
    custom_stopwords = set(stopword_file.read().splitlines())

tfidf = TfidfVectorizer(lowercase=True)


def remove_custom_stopwords(text):
    words = text.split()
    filtered_words = [word for word in words if word not in custom_stopwords]
    return ' '.join(filtered_words)


for data_directory in dataset_directories:
    document_texts = []
    for filename in os.listdir(data_directory):
        if filename.endswith('.txt'):
            file_path = os.path.join(data_directory, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
                document_texts.append(text)

    document_texts = [remove_custom_stopwords(text) for text in document_texts]

    all_document_texts.extend(document_texts)

tfidf_result = tfidf.fit_transform(all_document_texts)

joblib.dump(tfidf, 'tfidf_vectorizer.pkl')
joblib.dump(tfidf.vocabulary_, 'tfidf_vocabulary.pkl')

# # Print the vocabulary, if needed
# print('\nindexes numbers:')
# print(tfidf.vocabulary_)
#
# # Print the TF-IDF values, if needed
#print('\nTF-IDF values:')
#print(tfidf_result)
