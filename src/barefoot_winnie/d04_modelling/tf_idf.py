from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from barefoot_winnie.d04_modelling.feature_generator import FeatureGenerator


class TFIDF(FeatureGenerator):
    def __init__(self):
        FeatureGenerator.__init__(self)

    def fit_model(self, train_data: pd.Series, train_params=dict(), save_path=None):
        corpus = train_data.values.tolist()
        self.model = TfidfVectorizer(**train_params)

        # training the model
        self.model.fit(corpus)

        # saving the model
        if save_path is not None:
            self.save_model(save_path)

    def generate_features(self, intermediate_series: pd.Series, saved_model_path=None):
        """ Generates TF-IDF features on the bag of words
            intermediate_series: series of preprocessed questions/answers
            saved_model_path:  the path of the already trained TFIDF vectorizer
        """

        corpus = intermediate_series.values.tolist()

        if saved_model_path is not None:
            self.model = self.load_saved_model(saved_model_path)
        else:
            if self.model is None:  
                # If the model is not trained, trained using the available dataset. 
                # equivalent to the fit_transform
                self.fit_model(train_data=intermediate_series)

        return self.model.transform(corpus).todense()
