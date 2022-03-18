import sklearn.linear_model as linear_model

import sklearn.feature_extraction.text as fe_text
import sklearn.ensemble


class LogisticRegression:
    def __init__(self):
        self.vectorizer = fe_text.TfidfVectorizer()
        self.model = linear_model.LogisticRegression()


    def train(self, texts, labels):
        self.vectorizer.fit(texts)
        vecs = self.vectorizer.transform(texts)
        self.model.fit(vecs, labels)


    def predict(self, texts):
        vectors = self.vectorizer.transform(texts)
        return self.model.predict(vectors)


class RandomForest():
    def __init__(self):
        self.vectorizer = fe_text.TfidfVectorizer()
        self.model = sklearn.ensemble.RandomForestClassifier()

    def train(self, texts, labels):
        self.vectorizer.fit(texts)
        vecs = self.vectorizer.transform(texts)
        self.model.fit(vecs, labels)

    def predict(self, texts):
        vectors = self.vectorizer.transform(texts)
        return self.model.predict(vectors)