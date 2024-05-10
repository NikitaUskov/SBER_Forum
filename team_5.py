import pandas as pd
from joblib import load
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


def preprocess_text(text):
    words = nltk.word_tokenize(text)

    words = [lemmatizer.lemmatize(word.lower()) for word in words if word.lower() not in stop_words and word.isalpha()]
    return ' '.join(words)


if __name__ == "__main__":
    model_path = input("Введите путь к модели:\n")
    vector_path = input("Введите путь к векторизатору:\n")
    filename = input("Введите путь до файла:\n")

    data = pd.read_csv(filename)

    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('omw-1.4')

    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    data['Task_processed'] = data['Задача en'].apply(preprocess_text)
    data['Situation_processed'] = data['Обстановка en'].apply(preprocess_text)
    data['Optimum_processed'] = data['Оптимальный план en'].apply(preprocess_text)
    data['Predicted_processed'] = data['Предсказанный план'].apply(preprocess_text)

    tfidf_vectorizer = load(vector_path)
    combined_text = data['Situation_processed'] + " " + data['Predicted_processed']

    X = tfidf_vectorizer.transform(combined_text)

    clf = load(model_path)
    y_pred = clf.predict(X)
    (pd.DataFrame({"Успех предсказанного плана": y_pred})).to_csv("labels_5.csv", sep=",", index=False)
