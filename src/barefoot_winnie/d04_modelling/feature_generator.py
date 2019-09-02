import joblib


class FeatureGenerator:
    """ Generates features for a specific column in the dataframe. E.g. Question"""
    def __init__(self):
        self.model = None
    
    def generate_features(self, intermediate_series):
        return None

    def save_model(self, save_path):
        joblib.dump(self.model, save_path)

    def load_saved_model(self, model_path):
        model = joblib.load(model_path)
        return model
