import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from itertools import cycle

from barefoot_winnie.d00_utils.preprocessing import run_preprocessing_steps
from barefoot_winnie.d04_modelling.tf_idf import TFIDF
from barefoot_winnie.d04_modelling.word2vec_pretrained import Word2VecPreTrained


class NNExperiment:
    """This class contains the functionality of a nearest neighbor experiment"""
    def __init__(self, 
                 data_set: pd.DataFrame, train_ratio = 1,
                 num_neighbors=5,
                 feature_vectorizer=TFIDF(),
                 dist_metric='cosine',
                 eval_metric='edit',
                 preprocessing_steps=None,
                 save_results=True):

        self.num_neighbors = num_neighbors

        if preprocessing_steps is None:
            self.preprocessing_steps = ['denoise_text',
                                        'replace_contractions',
                                        'remove_stop_words',
                                        'stem_words',
                                        'lemmatize_words']
        self.df = data_set.reset_index(drop=True)  # raw data
        self.train_ratio = train_ratio

        # Feature Vector Creation
        self.vectorizer = feature_vectorizer
        self.dist_metric = dist_metric
        self.eval_metric = eval_metric
        self.results = {}
        self.save_results = save_results
        self.q_col = 'question'
        self.r_col = 'answer'
        self.train_msk = list()

    def setup_experiment(self, process_response=False):
        """ Prprocesses data, creates features and splits train and test sets"""

        self.df = self.df.dropna(subset=[self.q_col, self.r_col])

        # preprocess columns
        q_col_intermediate = run_preprocessing_steps(series=self.df[self.q_col],
                                                     steps=self.preprocessing_steps)

        # generate features
        q_col_features = self.vectorizer.generate_features(q_col_intermediate)
        q_col_features_df = pd.DataFrame(q_col_features, index=q_col_intermediate.index)

        if process_response:
            r_col_intermediate = run_preprocessing_steps(series=self.df[self.r_col],
                                                     steps=self.preprocessing_steps)
            r_col_features = self.vectorizer.generate_features(r_col_intermediate)
            r_col_features_df = pd.DataFrame(r_col_features)

        # train_test split
        if self.train_ratio < 1:
            self.train_msk = np.random.rand(len(q_col_features_df)) < self.train_ratio
            test_set = q_col_features_df[~self.train_msk]
            train_set = q_col_features_df[self.train_msk]
        else:
            # TODO: Handle the case where train_ratio is 1
            train_set = q_col_features_df
            test_set = None

        return {'train_set': train_set, 'test_set': test_set}

    def run(self):  
        exp_setup = self.setup_experiment()
        train_set = exp_setup['train_set']
        test_set = exp_setup['test_set']

        # Fitting the nearest neighbors to the train set
        neigh = NearestNeighbors(n_neighbors=self.num_neighbors, metric='cosine')
        neigh.fit(train_set)
        
        # Getting the k-nearest neighbors w.r.t the train set
        # If the test set is not set, the train set is used to return neighbors
        if test_set is not None:
            # these indices correspond to 0-indexed train_set
            dist, ind = neigh.kneighbors(test_set)
        else:
            dist, ind = neigh.kneighbors(train_set)
        self.package_results(ind, dist)

    def package_results(self, nn_indices, nn_distances):

        self.results = self.df[~self.train_msk]
        self.results = self.results.loc[self.results.index.repeat(self.num_neighbors)]

        self.results['nn_indices'] = nn_indices.flatten()
        self.results['nn_distances'] = nn_distances.flatten()
        responses = self.df[self.train_msk].iloc[list(nn_indices.flatten())][self.r_col]
        self.results['response'] = responses.values
        self.results['response_id'] = list(responses.index)

        rank_cycle = cycle(list(range(self.num_neighbors)))
        self.results['rank'] = [next(rank_cycle) for _ in range(len(self.results))]
        self.results['question_id'] = list(self.results.index)
        self.results['true_response'] = list(self.df[self.r_col].loc[list(self.results['question_id'])])


def run_w2v_experiment(data_set: pd.DataFrame, parameters: dict) -> dict:
    """Defines a W2V experiment. This function is used Model selection"""

    # Loading the pretrained file and defining the Feature Vector Object
    pretrained_vector_file = parameters['pretrained_vector_file']
    w2v = Word2VecPreTrained(pretrained_vector_file)

    # Experimental Setup
    setup = {
        'data_set': data_set,
        'num_neighbors': parameters['num_neighbors'],
        'train_ratio': parameters['train_ratio'],
        'feature_vectorizer': w2v,
        'dist_metric': parameters['dist_metric'],
        'eval_metric': parameters['eval_metric'],
        'preprocessing_steps': parameters['preprocessing_steps'],
        'save_results': parameters['save_results']
    } 

    # Running the experiment
    exp = NNExperiment(**setup)
    exp.run()

    return exp.results


def run_tfidf_experiment(data_set:pd.DataFrame, parameters: dict) -> dict:
    """Defines a TF_IDF/BOW experiment, This function is used Model selection"""

    tfidf = TFIDF()
    # Experimental Setup
    setup = {
        'data_set': data_set,
        'num_neighbors': parameters['num_neighbors'],
        'train_ratio': parameters['train_ratio'], 
        'feature_vectorizer': tfidf,
        'dist_metric': parameters['dist_metric'],
        'eval_metric': parameters['eval_metric'],
        'preprocessing_steps': parameters['preprocessing_steps'],
        'save_results': parameters['save_results']    
    } 

    # Running the experiment
    exp = NNExperiment(**setup)
    exp.run()

    return exp.results
