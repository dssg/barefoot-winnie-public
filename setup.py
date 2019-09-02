import nltk
import logging


"""Setting up logging"""
logging.basicConfig(filename='winnie.log', level=logging.INFO)

""" Downloading the nltk resources"""
logging.info('Setting up NLTK resouces...')

nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
