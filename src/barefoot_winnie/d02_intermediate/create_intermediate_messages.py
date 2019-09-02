import numpy as np


def create_intermediate_messages(raw_messages):

    intermediate_fb_messages = raw_messages.copy()
    intermediate_fb_messages.drop(
        columns=['id', 'user_id', 'created_at', 'updated_at', 'deleted_at', 'message_type', 'facebook_message_id'],
        inplace=True)
    intermediate_fb_messages['body'].replace('', np.nan, inplace=True)
    intermediate_fb_messages.dropna(subset=['body'], inplace=True)

    return intermediate_fb_messages

