import pandas as pd
import numpy as np


def restructure_intermediate_input(intermediate_input: pd.DataFrame,
                                   gap_between_conversations=10):

    # identify conversations
    intermediate_input.sort_values(by=['thread_id', 'message_sent_date'], inplace=True)
    intermediate_input['days_since_last_message'] = ((intermediate_input.message_sent_date
                                                      - intermediate_input.groupby('thread_id').message_sent_date.shift(1))
                                                     .dt.days
                                                     )
    intermediate_input['new_conversation_flag'] = (intermediate_input.days_since_last_message.isnull() |
                                                   (intermediate_input.days_since_last_message >= gap_between_conversations))
    intermediate_input['conversation_id'] = intermediate_input.groupby('thread_id').new_conversation_flag.cumsum().astype(int)

    # identify speakers
    intermediate_input['new_speaker_flag'] = (intermediate_input.groupby(['thread_id', 'conversation_id'])['from'].shift(1).isnull() |
                                              (intermediate_input['from'] !=
                                               intermediate_input.groupby(['thread_id', 'conversation_id'])['from'].shift(1)))
    intermediate_input['speaker_id'] = intermediate_input.groupby(['thread_id', 'conversation_id']).new_speaker_flag.cumsum()

    # concatenate conversations that belong together
    msg_concat = (intermediate_input
                  .groupby(['thread_id', 'conversation_id', 'speaker_id'])
                  ['body']
                  .apply('\\\\\\'.join)
                  .reset_index())

    # use the meta data from the first message of the speaker
    meta_data = intermediate_input.loc[intermediate_input.new_speaker_flag,
                                       ['thread_id', 'from', 'message_sent_date',
                               'days_since_last_message', 'conversation_id', 'speaker_id','id']]

    msg_concat = pd.merge(left=msg_concat,
                          right=meta_data,
                          how='left',
                          on=['thread_id', 'conversation_id', 'speaker_id'])

    # for the first message in a thread, fill the number of days since last message with 0.
    msg_concat['days_since_last_message'] = msg_concat['days_since_last_message'].fillna(0)

    msg_concat['type'] = np.where(msg_concat['from'] == 'Barefoot Lawyers- Uganda',
                                  'answer',
                                  'question')

    msg_concat['question_flag'] = msg_concat['type'] == 'question'
    msg_concat['interaction_counter'] = (msg_concat
                                         .groupby(['thread_id', 'conversation_id'])
                                         .question_flag
                                         .cumsum()
                                         .astype(int))

    msg_concat['interaction_id'] = (msg_concat['thread_id']
                                    + '_'
                                    + msg_concat['conversation_id'].astype(str)
                                    + '_'
                                    + msg_concat['interaction_counter'].astype(str))

    id_cols = ['interaction_id']
    question_cols = {'thread_id': 'thread_id',
                     'id':'id',
                     'conversation_id': 'num_of_conversations',
                     'speaker_id': 'num_of_messages',
                     'body': 'question',
                     'from': 'beneficiary_name',
                     'message_sent_date': 'question_asked_time',
                     'days_since_last_message': 'days_since_last_message'}
    answer_cols = {'body': 'answer',
                   'message_sent_date': 'answer_given_time',
                   'days_since_last_message': 'days_taken_to_respond'}
    question_df = (msg_concat.loc[msg_concat.question_flag,
                                  list(question_cols.keys()) + id_cols]
                   .rename(columns=question_cols))
    answer_df = (msg_concat.loc[~msg_concat.question_flag,
                                list(answer_cols.keys()) + id_cols]
                 .rename(columns=answer_cols))

    # this drops interactions where BFL reached out without a question being asked by beneficiaries
    intermediate_restructured = pd.merge(left=question_df,
                                         right=answer_df,
                                         on=id_cols,
                                         how='left')

    return intermediate_restructured
