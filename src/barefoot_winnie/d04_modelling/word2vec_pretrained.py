from gensim.models import KeyedVectors
import numpy as np
from barefoot_winnie.d04_modelling.feature_generator import FeatureGenerator


class Word2VecPreTrained(FeatureGenerator):
    """ Class for creating average pretrained W2V features. Using google's pretrained vectors"""
    def __init__(self, pretrained_vectors_path, embedding_len=300):
        """
        :param intermediate_series: Series to get the embedding
        :param pretrained_vectors_path: Path to the downloaded pretrained vectors
        :param embedding_len: Vector size for each word.Dependent on the pretrained library. For google, its 300
        """
        FeatureGenerator.__init__(self)
        self.pretrained_vectors_path = pretrained_vectors_path
        self.embedding_len = embedding_len  # Google pre-trained embedding length is 300

    def fit_model(self, train_data=None, train_params=dict(), save_path=None):
        """ Loads the pretrained Vectors and saves the model"""
        self.model = KeyedVectors.load_word2vec_format(self.pretrained_vectors_path, binary=True)

        # Saving the model
        if save_path is not None:
            self.save_model(save_path)

    def get_document_embedding(self, document):
        """ Converts a document to a numeric vector of length 300 (Google Pre-trained len)
            Return: Mean of all word embeddings in the questions
        """
        words = document.split()
        vects = np.zeros((len(words), self.embedding_len))

        # Computing the word embedding for each word.
        # If the word is not found in the dictionary, vector of zeros is used
        for i, word in enumerate(words):
            try:
                vects[i] = self.model.get_vector(word)
            except KeyError:
                continue

        return vects.mean(axis=0)

    def generate_features(self, intermediate_series, saved_model_path=None):
        """ Get the document embeddings for all questions/answers. Loads a model that been saved.
            If not saved, fits the model 
            return: 2D numpy array (num_questions x word_embedding_length)
        """

        if saved_model_path is not None:
            self.model = self.load_saved_model(saved_model_path)

        else:
            if self.model is None:
                self.fit_model()

        features = intermediate_series.apply(lambda x: self.get_document_embedding(x))
        features = np.column_stack(features.values).T

        # Removing NaN values in the data 
        features = np.nan_to_num(features)
        return features



