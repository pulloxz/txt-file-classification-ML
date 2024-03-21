import os
import joblib
from data_processing import remove_custom_stopwords
from handlingmismatch import trim_or_pad_features


def classify_files_in_folder(input_folder, output_folder):
    svm_model = joblib.load('svm_model.pkl')

    tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')
    vocabulary = joblib.load('tfidf_vocabulary.pkl')
    tfidf_vectorizer.vocabulary_ = vocabulary

    label_to_class = {0: 'sport', 1: 'business', 2: 'entertainment', 3: 'food', 4: 'technology',
                      5: 'space', 6: 'politics', 7: 'medical', 8: 'historical', 9: 'graphics',
                      10: 'others'}

    for class_name in label_to_class.values():
        class_output_folder = os.path.join(output_folder, class_name)
        os.makedirs(class_output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.endswith('.txt'):
            file_path = os.path.join(input_folder, filename)

            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()

            text = remove_custom_stopwords(text)

            text_tfidf = tfidf_vectorizer.transform([text])

            X_train_tfidf = joblib.load('tfidf_result.pkl')
            text_tfidf = trim_or_pad_features(text_tfidf, X_train_tfidf.shape[1])

            predicted_label = svm_model.predict(text_tfidf)

            predicted_class = label_to_class[predicted_label[0]]

            output_class_folder = os.path.join(output_folder, predicted_class)
            if not os.path.exists(output_class_folder):
                os.makedirs(output_class_folder)

            output_file_path = os.path.join(output_class_folder, filename)
            os.replace(file_path, output_file_path)


if __name__ == "__main__":
    input_folder_path = '/Users/fatimamuhammed/Downloads/archive/bbc/tech'
    output_folder_path = '/Users/fatimamuhammed/desktop'
    classify_files_in_folder(input_folder_path, output_folder_path)
