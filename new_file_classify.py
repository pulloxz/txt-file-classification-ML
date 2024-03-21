import joblib
from data_processing import remove_custom_stopwords
from handlingmismatch import trim_or_pad_features

svm_model = joblib.load('svm_model.pkl')

tfidf = joblib.load('tfidf_vectorizer.pkl')

vocabulary = joblib.load('tfidf_vocabulary.pkl')

tfidf.vocabulary_ = vocabulary
X_train_tfidf = joblib.load('tfidf_result.pkl')

# any instance that doesn't belong to one of the specific classes (0 to 9) will be categorized under the 'others' class
label_to_class = {0: 'sport', 1: 'business', 2: 'entertainment', 3: 'food', 4: 'technology', 5: 'science', 6: 'politics',
                  7: 'medical', 8: 'historical', 9: 'graphics', 10: 'others'}

new_file_path = '/Users/fatimamuhammed/Desktop/v.rtf'#edit path as needed
with open(new_file_path, 'r', encoding='utf-8') as file:
    new_text = file.read()

new_text = remove_custom_stopwords(new_text)
new_text_tfidf = tfidf.transform([new_text])


new_text_tfidf = trim_or_pad_features(new_text_tfidf, X_train_tfidf.shape[1])

predicted_label = svm_model.predict(new_text_tfidf)

predicted_class = label_to_class[predicted_label[0]]

print('Predicted Class:', predicted_class)
