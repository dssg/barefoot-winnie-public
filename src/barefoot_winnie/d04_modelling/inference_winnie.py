from sklearn.neighbors import NearestNeighbors
import pandas as pd
import numpy as np

from barefoot_winnie.d00_utils.preprocessing import run_preprocessing_steps
from barefoot_winnie.d00_utils.yaml_utils import get_path_catalog
from barefoot_winnie.d00_utils.yaml_utils import get_parameter_yaml
from barefoot_winnie.d04_modelling.tf_idf import TFIDF
from barefoot_winnie.d04_modelling.word2vec_pretrained import Word2VecPreTrained


def inference_winnie(question_data: pd.DataFrame,
                     feature_type='tfidf',
                     distance_metric='cosine',
                     num_neighbors=5):
    """ Estimated and returns a set of candidate responses to a new question(s)
    :param question_data: Text data and meta data for the new question
    :param feature_type: Type of features created, has to match the saved features
    :param distance_metric: distance metric used in the nearest neighbor method
    :param num_neighbors: Number of candidate responses to return
    :return: a pandas dataframe with candidate responses
    """

    # Loading Saved Model files
    saved_model_path = get_path_catalog('trained_model')
    numeric_file_path = get_path_catalog('model_numeric_vectors')
    raw_text_path = get_path_catalog('model_raw_text')
    numeric_vectors = pd.read_parquet(numeric_file_path, engine='pyarrow')
    raw_text = pd.read_parquet(raw_text_path, engine='pyarrow')

    # prepare question for inference
    train_winnie_settings = get_parameter_yaml('train_winnie_settings')
    if feature_type == 'tfidf':
        feature_vectorizer = TFIDF()

    elif feature_type == 'w2v':
        w2v_pretrained_file = train_winnie_settings['w2v_pretrained_file']
        feature_vectorizer = Word2VecPreTrained(w2v_pretrained_file)

    else:
        # Ensuring the feature vectorizer is intialized to TFIDF, in case of mistakes in parameters
        feature_vectorizer = TFIDF()

    # Note: this is the column name used in the MySQL database.
    question_series = question_data['consultation_highlights']
    question_series = question_series.dropna()

    preprocessing_steps = train_winnie_settings['preprocessing_steps']
    questions_preprocessed = run_preprocessing_steps(series=question_series, steps=preprocessing_steps)

    questions_preprocessed_features = feature_vectorizer.generate_features(intermediate_series=questions_preprocessed,
                                                                           saved_model_path=saved_model_path)

    # Set up Nearest Neighbors
    nearest_neighbor_model = NearestNeighbors(n_neighbors=num_neighbors, metric=distance_metric)
    nearest_neighbor_model.fit(numeric_vectors)

    # Finding the nearest neighbors
    dist, ind = nearest_neighbor_model.kneighbors(questions_preprocessed_features)

    result = []
    for question_counter in range(len(question_data)):
        recommendation_df = pd.DataFrame({'response_rank': list(np.arange(1, num_neighbors + 1)),
                                          'recommended_response': (raw_text['answer']
                                                                   .iloc[ind[question_counter]]
                                                                   .tolist()),
                                          'case_id': question_data['id'].iloc[question_counter]})
        result.append(recommendation_df)

    result = pd.concat(result, ignore_index=True)

    return result
