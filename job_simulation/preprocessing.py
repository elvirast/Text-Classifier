import re
from typing import Iterable, Generator
import nltk.corpus
import unicodedata
import contractions
import nltk
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()


USEFUL_NLTK_STOPWORDS = set(['nor',
                             'no',
                             'not',
                             'nothing',
                             'neither',
                             'never',
                             'none',
                             'up',
                             'down',
                             'latency',
                             'slot',
                             'standby',
                             'left',
                             'right',
                             'ise'])
NLTK_STOPWORDS = set(list(nltk.corpus.stopwords.words('english'))) - USEFUL_NLTK_STOPWORDS


def remove_urls(text: str) -> str:
    return re.sub(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*', '', text, flags=re.MULTILINE)


def remove_emails(text: str) -> str:
    return re.sub(r'\S*@\S*\s?', ' ', text)


def remove_multi_whitespaces(text: str) -> str:
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text


def remove_times_and_dates(text: str) -> str:
    # Remove time formats Ex:  05:09:09
    text = re.sub(r'\s\d{1,2}:\d{1,2}:\d{1,2}', ' ', text)

    # Remove date formats Ex:  05/09 ,   05/09/2014 05-09-2014 05-Jan-2014
    text = re.sub(r'\s\d{1,4}/\d{1,4}/\d{1,4}', ' ', text)
    text = re.sub(r'\s\d{1,4}-\d{1,4}-\d{1,4}', ' ', text)
    text = re.sub(r'\s\d{1,2}/\d{1,2}', ' ', text)
    # text = re.sub(r'\d{1,4}-\w{1,4}-\d{1,4}', ' ', text)
    # Remove time formats Ex:  05:09
    text = re.sub(r'\s\d{1,2}:\d{1,2}', ' ', text)

    # Remove date formats Ex:  05-09
    text = re.sub(r'\s\d{1,2}-\d{1,2}', ' ', text)
    return text


def remove_punctuation(text: str) -> str:
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    return text


def remove_repetition(text: str) -> str:
    # Remove multiple instance of the repeating characters -=/_ Ex: Replace observation==== with observation
    regex = re.compile(r'(\=\=+)|(\/\/+)|(\_\_+)|(\-\-+)')
    text = re.sub(regex, '', text)

    # Remove repeating words fan fan tay = fan tray
    text = re.sub(r'\b(\w+)( \1\b)+', r'\1', text)
    return text


def customize_wf_anonymization(text: str) -> str:
    text = re.sub('wells fargo', 'wells_fargo', text)

    text = re.sub(r'x{1,9}\/x{1,9}\/x{1,9}', 'xxxx', text)
    # text = re.sub(r'xx/xx/xx', 'xx', text)

    # text = re.sub(r'xx/xx/xxxx', 'xx', text)
    text = re.sub(r'x{1,9}\/x{1,9}\/\d{1,4}', 'xxxx', text)
    text = re.sub(r' \{\$\d{1,9}.\d{1,9}\}', ' xxxx', text)

    text = re.sub(r'\d{1,4}', 'xxxx', text)
    return text


def preprocess_text(text: str) -> str:
    """
    Normalizes the text, the order of the operations is important
    """

    text = text.lower()

    text = remove_urls(text)
    text = remove_emails(text)
    text = remove_times_and_dates(text)

    text = customize_wf_anonymization(text)

    text = remove_punctuation(text)
    text = remove_multi_whitespaces(text)

    text = remove_repetition(text)

    # Remove special characters except /_%-
    # text = re.sub('[-]','', text)
    text = re.sub('[^a-zA-Z0-9?/_%\n-]', ' ', text)

    text = '' if text == 'none' else text.strip()

    return text


def expand_contractions(text: str) -> str:
    return contractions.fix(text)


def remove_accented_chars(text: str) -> str:
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    return text


def preprocess_text_advanced(text: str, stem: bool = True) -> str:

    # remove extra newlines (often might be present in really noisy text)
    text = text.translate(text.maketrans("\n\t\r", "   "))

    text = remove_accented_chars(text)
    text = expand_contractions(text)

    text = ' '.join([word for word in text.split() if word not in USEFUL_NLTK_STOPWORDS and word != ''])
    if stem:
        # document = ' '.join([stemmer.stem(word) for word in document.split()])
        text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()])

    # remove extra whitespace
    text = re.sub(' +', ' ', text)
    text = text.strip()

    return text


def preprocess_texts(texts: Iterable[str]) -> Generator[str, None, None]:
    for text in texts:
        text = preprocess_text(text)
        text = preprocess_text_advanced(text)
        yield text

