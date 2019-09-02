from barefoot_winnie.d04_modelling.inference_winnie import inference_winnie


def generate_responses(case_data, num_neighbors=5):
    """ Wraps inference, Takes the case data (text and labels) and calls inference function of Winnie
    :param case_data: Pandas dataframe with question text and labels
    :param num_neighbors: Number of candidate responses to be returned
    :return: Pandas dataframe with he recommended response, response_rank and the case_id
    """

    responses_df = inference_winnie(case_data, num_neighbors=num_neighbors)

    dtype_dict = {'case_id': int,
                  'response_rank': int,
                  'recommended_response': str}

    for col, type_choice in dtype_dict.items():
        responses_df[col] = responses_df[col].astype(type_choice)

    return responses_df
