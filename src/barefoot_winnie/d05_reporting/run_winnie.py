from barefoot_winnie.d00_utils.fetch_case import fetch_case
from barefoot_winnie.d00_utils.save_responses import save_responses
from barefoot_winnie.d05_reporting.recommendation_generator import generate_responses


def run_winnie(case_id, num_neighbors=5):
    """ Runs the inference pipeline
    :param case_id: id of the case/inquiry that is being answered by the lawyer
    :param num_neighbors: number of answers to return per question
    :return: A flag indicating success
    """

    # Fetching the data for the case from the MySQL database
    case_data = fetch_case(case_id)

    # Runs Winnie to gets the candidate responses for the question
    case_responses = generate_responses(case_data, num_neighbors=num_neighbors)

    # Writes the responses to the MySQL database
    save_responses(case_responses)

    return True
