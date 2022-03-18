import emoji
import re
import translators as ts


def filter_wishes(df):
    filter_test = (df['title'] != 'Test') & (df['title'] != 'тест') & (
                df['title'] != 'Тест')
    filter_test = filter_test & (df['title'] != '1') | (df['link'].notnull())
    filter_test = filter_test & (df['title'] != '~')
    filter_test = filter_test & (df['link'] != 'ohmywishes.com')
    filter_test = filter_test & (df['title'] != 'Классный подарок')
    return df[filter_test]


def strip_emoji(text):
    new_text = re.sub(emoji.get_emoji_regexp(), r"", text)
    return new_text


def wishlist_text(nlp, text, names=[]):
    doc = nlp(text)

    name_entities = [ent.text for ent in doc.ents if ent.label_ == 'PER']
    if name_entities:
        return ''

    words = []
    for token in doc:
        if token.lemma_.lower() in names:
            return ''

        if (token.is_stop or token.lemma_ in nlp.Defaults.stop_words
                or token.is_space
                or token.is_punct):
            continue
        words.append(token.lemma_)

    result = ' '.join(words).lower()

    result = result.replace('новый год', '')
    result = result.replace('день рождения', '')
    result = result.replace('день рождение', '')
    result = result.replace('тайный санта', '')
    result = result.strip()

    if  result.isdigit():
        return ''

    return result.strip()


def process_texts(nlp, texts):
    result_texts = []
    for text in texts:
        words = [token.lemma_.lower() for token in nlp(text) if
                 not token.is_stop and not token.is_punct and token.is_alpha]
        result_texts.append(' '.join(words))
    return result_texts


def clear_text(text):
    text = strip_emoji(text.lower())
    return text.strip()


def ru_text(text):
    not_to_translate = ['apple', 'pandora', 'Baby shower', 'ozon', 'amazon',
                        'hema', 'pat mcgrath', 'shein', 'wildberries', 'leta',
                        'dyson']

    words = text.split()

    for word in words:
        if word in not_to_translate:
            return text

    return ts.google(text, to_language='ru').lower()
