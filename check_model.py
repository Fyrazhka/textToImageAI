import joblib
import os
from tensorflow.keras.models import load_model

project_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(project_dir, '..', 'data')
vectorizer = joblib.load(os.path.join(data_dir, 'tfidf_vectorizer.pkl'))


model = load_model(os.path.join(data_dir, 'sentiment_model_nn.h5'))
vectorizer = joblib.load((os.path.join(data_dir,'tfidf_vectorizer.pkl')))


def predict_sentiment_nn(review_text):
    review_tfidf = vectorizer.transform([review_text]).toarray()
    prediction = model.predict(review_tfidf)
    return "Positive" if prediction[0] > 0.5 else "Negative"


new_review = """
Gómez Pereira is the responsible for some of the most horrible comedies of latest Spanish cinema (just take a look at his curriculum vitae), so I didn't expect that much of "Cosas Que Hacen..."... In fact I don't know why in the world did I decide to watch it. Anyway, I just did... And what a surprise. It looks that Gómez Pereira has finally matured and now he's capable of making a good movie. He's last work deals with the midlife crisis, the disappointing, and the seeking for a second chance after you've ruined it all. The last half hour of the movie (the more dramatic) is the best part, and it just makes worth watching the film. Also we have Eduard Fernandez playing the main role, and I keep on thinking he's the worest actor of his generation (by far).<br /><br />*
"""
print(f"Review: {new_review}")
print(f"Prediction: {predict_sentiment_nn(new_review)}")
