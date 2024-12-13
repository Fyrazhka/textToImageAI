import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import os
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

dataset_path = os.path.join(os.path.dirname(__file__))


def load_reviews_from_folder(folder_path, label):
    reviews = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as file:
                reviews.append([file.read(), label])
    return reviews


pos_reviews = load_reviews_from_folder(os.path.join(dataset_path, 'train', 'pos'), 1)
neg_reviews = load_reviews_from_folder(os.path.join(dataset_path, 'train', 'neg'), 0)


reviews = pos_reviews + neg_reviews
df = pd.DataFrame(reviews, columns=['review', 'sentiment'])


print(df.head())


X = df['review']
y = df['sentiment']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

X_train_tfidf = X_train_tfidf.toarray()
X_test_tfidf = X_test_tfidf.toarray()


model = Sequential()


model.add(Dense(128, input_dim=X_train_tfidf.shape[1], activation='relu'))


model.add(Dropout(0.5))


model.add(Dense(64, activation='relu'))


model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])


model.fit(X_train_tfidf, y_train, epochs=5, batch_size=64, validation_data=(X_test_tfidf, y_test))


loss, accuracy = model.evaluate(X_test_tfidf, y_test)
print(f'Accuracy: {accuracy * 100:.2f}%')


model.save('sentiment_model_nn.h5')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')

