from bs4 import BeautifulSoup
import re
from nltk.stem import LancasterStemmer, WordNetLemmatizer
import contractions
from nltk.corpus import stopwords
import string
import num2words
import nltk
import pandas as pd
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize


def strip_html_tags(text):
    """ removes HTML tags in the text"""
    soup = BeautifulSoup(text, "html.parser").text
    return soup


def strip_urls(text):
    """Strips any URLs in the text (has to be prepended by http/www)"""
    t = re.sub(r'(http|www)\S+', '', text)
    return t


def remove_between_square_brackets(text):
    """ Removes any text placed between square brackets"""
    return re.sub('\[[^]]*\]', '', text)


def denoise_text(text):
    """ combines all the denoising steps"""
    text = strip_html_tags(text)
    text = strip_urls(text)
    text = remove_between_square_brackets(text)

    return text


def stem_words(text):
    """ combines the different forms of the verbs/adverbs/adjectives"""
    text = text.split()
    try:
    	stemmer = LancasterStemmer()
    except LookupError:
        nltk.download('wordnet')

    stems = list()
    for word in text:
        stem = stemmer.stem(word)
        stems.append(stem)
    return ' '.join(stems)


def lemmatize_words(text):
    """ converts the word into its root form"""
    try:
        lemmatizer = WordNetLemmatizer()
    except LookupError:
        nltk.download('wordnet')
        lemmatizer = WordNetLemmatizer()
    
    lemmas = list()
    for word in text.split():
        lemma = lemmatizer.lemmatize(word, pos='v')
        lemmas.append(lemma)

    return ' '.join(lemmas)


def remove_stop_words(text):
    """ Remove stop words from raw text"""
    try:
        stop_words = set(stopwords.words('english'))
    except LookupError:
        nltk.download('stopwords')
        stop_words = set(stopwords.words('english'))  

    return ' '.join([item for item in text.split() if item not in stop_words])


def remove_punctuation(text):
    """Remove all punctuation in text and replace with whitespace"""
    translator = str.maketrans(string.punctuation,
                               ' ' * len(string.punctuation))
    return text.translate(translator)


def replace_numbers_with_words(text):
    """convert numbers into words: e.g. 1 to one"""
    return ' '.join([num2words.num2words(s) if s.isdigit() else s for s in text.split()])


def remove_numbers(text):
    """replace numbers with whitespace"""
    return ' '.join(['' if s.isdigit() else s for s in text.split()])


def filter_pos_tags(text, list_pos_tags=None):
    """retains the defined POS tags in text and removes the rest"""
    if list_pos_tags is None:
        list_pos_tags = ['NN', 'NNP', 'NNS', 'NNPS', 'VB', 'VBD' 'VBG', 'VBN', 'VBP', 'VBZ']

    list_retained = [word for word,pos in pos_tag(word_tokenize(text)) if pos in list_pos_tags]

    return ' '.join(list_retained)


def remove_custom_stop_words(text, stop_word_list=None):
    """removing words from the text that occur very often in data that don't carry much info"""
    if stop_word_list is None:
        stop_word_list = \
        ['hi','hello', 'thanks','thanx', 'dear', 
        'thank you','thank', 'you', 'barefoot', 'barefootlaw', 'foot', 'please','help','ok', 
        'law','lawyer','yer', 'ts','uganda', 'need']

    text_words = text.split()
    filtered_word_list = [word for word in text_words if word.lower() not in stop_word_list]

    return ' '.join(filtered_word_list)


def run_preprocessing_steps(series: pd.Series, steps: list=None) -> pd.Series:
    """ Runs the preprocesing steps sequentially and returns the preprocessed series"""

    pre_process_dict = {'remove_punctuation': remove_punctuation,
                        'denoise_text': denoise_text,
                        'remove_numbers': remove_numbers,
                        'replace_contractions': contractions.fix,
                        'remove_stop_words': remove_stop_words,
                        'stem_words': stem_words,
                        'lemmatize_words': lemmatize_words,
                        'filter_pos_tags': filter_pos_tags,
                        'remove_custom_stop_words': remove_custom_stop_words}

    if steps is None:
        steps = ['denoise_text',
                 'remove_punctuation',
                 'remove_numbers',
                 'filter_pos_tags',
                 'replace_contractions',
                 'remove_stop_words',
                 'remove_custom_stop_words',
                 'stem_words',
                 'lemmatize_words']

    # Converting all the entries in the series to lower case.
    series = series.str.lower()

    for step in steps:
        series = series.apply(lambda x: pre_process_dict[step](x))

    return series

