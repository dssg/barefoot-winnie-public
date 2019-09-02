import pandas as pd

from barefoot_winnie.d00_utils.preprocessing import run_preprocessing_steps
from barefoot_winnie.d04_modelling.tf_idf import TFIDF
from barefoot_winnie.d04_modelling.word2vec_pretrained import Word2VecPreTrained


def train_winnie(train_data: pd.DataFrame, train_winnie_settings: dict) -> list:
    """ Trains Winnie and returns the three components of Winnie
    :param train_data: A pandas dataframe with question-answer pairs. Should contain columns 'question' and 'answer'
    :param train_winnie_settings: Train hyper-parameter settings from the parameters.yml file
    :return: 1. Trained feature vectorizer, 2. Numeric vectors of the train data, 3. Raw text data of the train data.
    """

    # Reading in the parameters for training
    feature_type = train_winnie_settings['feature_type']
    preprocessing_steps = train_winnie_settings['preprocessing_steps']
    train_params = train_winnie_settings['train_params']
    w2v_pretrained_file = train_winnie_settings['w2v_pretrained_file']

    # Setting the vectorizer
    if feature_type == 'tfidf':
        feature_vectorizer = TFIDF()

    elif feature_type == 'w2v':
        feature_vectorizer = Word2VecPreTrained(w2v_pretrained_file)

    else:
        # Ensuring the feature vectorizer is intialized to TFIDF, in case of mistakes in parameters
        feature_vectorizer = TFIDF()

    # Dropping unpolulated question answer pairs
    train_data = train_data.dropna(subset=['question', 'answer'])

    question = train_data['question']
    answer = train_data['answer']

    questions_preprocessed = run_preprocessing_steps(series=question, steps=preprocessing_steps)

    # Fitting and saving the model
    feature_vectorizer.fit_model(train_data=questions_preprocessed,
                                 train_params=train_params)

    # formatting output
    feature_matrix = feature_vectorizer.generate_features(questions_preprocessed)
    feature_dataframe = pd.DataFrame(feature_matrix, index=questions_preprocessed.index)
    feature_dataframe.columns = feature_dataframe.columns.astype(str)
    text_dataframe = pd.DataFrame({'question': question,
                                   'answer': answer})

    return [feature_vectorizer.model, feature_dataframe, text_dataframe]
