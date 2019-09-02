from sklearn.feature_extraction.text import CountVectorizer
from barefoot_winnie.d03_primary.feature_generator import FeatureGenerator


class BOW(FeatureGenerator):
    def __init__(self):
        FeatureGenerator.__init__(self)

    def generate_features(self, intermediate_series):
        corpus = intermediate_series.values.tolist()

        vectorizer = CountVectorizer()
        return vectorizer.fit_transform(corpus).todense()
