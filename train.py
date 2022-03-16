import pandas as pd
import spacy
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import sklearn

import wishlist_groups


nlp = spacy.load('ru_core_news_md')

def process_texts(nlp, texts):
    result_texts = []
    for text in texts:
        words = [token.lemma_.lower() for token in nlp(text) if
                 not token.is_stop and not token.is_punct and token.is_alpha]
        result_texts.append(' '.join(words))
    return result_texts


def train_logistic_regression(vectorizer, texts, labels):
    vecs = vectorizer.transform(texts)

    model = LogisticRegression()
    model.fit(vecs, labels)
    return model


def train_random_forest(vectorizer, texts, labels):
    vecs = vectorizer.transform(texts)

    model = RandomForestClassifier()
    model.fit(vecs, labels)
    return model


def validate(model, vectorizer, texts, labels):
    vecs = vectorizer.transform(texts)
    y_pred = model.predict(vecs)
    return sklearn.metrics.classification_report(labels, y_pred)


wishlists = pd.read_csv('new_wishlists.csv', usecols=['id', 'title', 'new'])
wishlist_wish = pd.read_csv('data/db/wish_list_wishes.csv', usecols=['wish_list_id', 'wish_id'])
wishes = pd.read_csv('data/db/wishes.csv', usecols=['id', 'title', 'description', 'link'])

wishlist_groups = wishlist_groups.groups

def group_wish_titles(group_list):
    wishlist_ids = wishlists[wishlists['new'].isin(group_list)]['id'].to_list()
    wishes_ids = wishlist_wish[wishlist_wish['wish_list_id'].isin(wishlist_ids)]['wish_id'].to_list()

    group_wishes = wishes[wishes['id'].isin(wishes_ids)]
    return group_wishes['title'].unique()

texts = []
labels = []

for label in wishlist_groups:
    group_wishes = group_wish_titles(wishlist_groups[label])
    texts.extend(group_wishes)
    labels.extend([label] * len(group_wishes))

texts = process_texts(nlp, texts)

X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size = 0.3, random_state = 42, stratify=labels)
text_train, text_test, y_train, y_test = train_test_split(texts, labels, test_size = 0.3, random_state = 42, stratify=labels)

vectorizer = TfidfVectorizer()
vectorizer.fit_transform(text_train)

log_reg = train_logistic_regression(vectorizer, X_train, y_train)

print('Logistic Regression')
print(validate(log_reg, vectorizer, X_test, y_test))

random_forest = train_random_forest(vectorizer, X_train, y_train)
print('Random Forest')
print(validate(random_forest, vectorizer, X_test, y_test))










