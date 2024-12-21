import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Наши данные / Our dataset
dataset = pd.read_csv('data/fake_news.csv')

# RU: Проверим на пропущенные значения
# В наборе данных есть несколько столбцов title, text, label.
# EN: Check for missing values
# The dataset has several columns title, text, label.
print(dataset.head())

# RU:# Нам нужен столбец текста для обучения модели.
# EN:# We need the text column to train the model.
X = dataset['text']
# RU: И столбец label - для того, чтобы понять, является ли новость обманом или нет.
# EN: And a label column to see if the news is fake or not.
y = dataset['label']
# RU: Выше используем большую X и маленькую y для того, чтобы подчеркнуть, что X имеет большую размерность, чем y.
# X станет матрицей признаков, а y - вектор который просто содержит метки для этих признаков(Fake/Real).
# EN:Above we use large X and small y to emphasize that X has a higher dimensionality than y.
# X - will be the feature matrix and y will be a vector that simply contains the labels for those features (Fake/Real).

# RU: Разделим данные на обучающую и тестовую выборки.
# EN: Split the data into training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# RU: Преобразование текстовых данных в числовые с помощью TfidfVectorizer
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
# RU: Преобразуем текстовые данные в векторные представления.
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# RU: Создание и обучение модели PassiveAggressiveClassifier
model = PassiveAggressiveClassifier(max_iter=50, random_state=42)
# Модель начинаем обучатся и настраивать веса.
# Модель пытается наиболее подходящие веса для каждого признака.
model.fit(X_train_tfidf, y_train)

# Предсказания
y_pred = model.predict(X_test_tfidf)
# Проверка точности модели
accuracy = accuracy_score(y_test, y_pred)
print(f'Точность модели: {accuracy * 100:.2f}%')

# Визуализация матрицы ошибок
c_m = confusion_matrix(y_test, y_pred, labels=['FAKE', 'REAL'])
ConfusionMatrixDisplay(c_m, display_labels=['FAKE', 'REAL']).plot(cmap='viridis')
plt.title('Матрица ошибок')
plt.show()

# Графическое представление важности признаков(слов)
feature_names = vectorizer.get_feature_names_out()
# Получаем веса(признаки) для каждого слова из нашего текстового признака.
coefs = model.coef_[0]
# Сортируем веса по возрастанию и берем первые 10 индексов.
indices = np.argsort(coefs)[-10:]
plt.figure(figsize=(10, 6))

# В цикле мы получаем каждое слово, Ось y. coefs[indices] - получаем веса для текущего слова, Ось -x.
plt.barh([feature_names[i] for i in indices], coefs[indices], color='blue')
plt.title('Топ 10 слов с наибольшим вкладом в модель')
plt.xlabel('Важность каждого слова')
plt.ylabel('Слова')
plt.show()
