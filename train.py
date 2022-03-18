import pandas as pd
import spacy
from sklearn.model_selection import train_test_split
import wishlist_groups
import helpers
import models
import sklearn.metrics


wishlists = pd.read_csv('new_wishlists.csv', usecols=['id', 'title', 'new'])
wishlist_wish = pd.read_csv('data/db/wish_list_wishes.csv', usecols=['wish_list_id', 'wish_id'])
wishes = pd.read_csv('data/db/wishes.csv', usecols=['id', 'title', 'description', 'link'])

wishlist_groups = wishlist_groups.groups

def group_wish_titles(group_list):
    wishlist_ids = wishlists[wishlists['new'].isin(group_list)]['id'].to_list()
    wishes_ids = wishlist_wish[wishlist_wish['wish_list_id'].isin(wishlist_ids)]['wish_id'].to_list()

    group_wishes = wishes[wishes['id'].isin(wishes_ids)]
    return group_wishes['title'].unique()


def check_model(model, model_name):
    model.train(X_train, y_train)
    pred_y = model.predict(X_test)
    print(model_name)
    print(sklearn.metrics.classification_report(y_test, pred_y))


texts = []
labels = []

for label in wishlist_groups:
    group_wishes = group_wish_titles(wishlist_groups[label])
    texts.extend(group_wishes)
    labels.extend([label] * len(group_wishes))

nlp = spacy.load('ru_core_news_md')
texts = helpers.process_texts(nlp, texts)

X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size = 0.2, random_state = 42, stratify=labels)

check_model(models.LogisticRegression(), 'Logistic Regression')
check_model(models.RandomForest(), 'Random Forest')




