import numpy as np


def create_intermediate_received_sms(raw_received_sms):

    intermediate_received_sms = raw_received_sms.copy()
    intermediate_received_sms.drop(
        columns=['created_at', 'updated_at', 'recipient', 'message_type', 'is_read'],
        inplace=True)

    intermediate_received_sms.rename(
        columns={'message': 'body', 'date_sent': 'message_sent_date'},
        inplace=True)
    intermediate_received_sms['body'].replace('', np.nan, inplace=True)
    intermediate_received_sms.dropna(subset=['body'], inplace=True)

    return intermediate_received_sms


def create_intermediate_sent_sms(raw_sent_sms):

    intermediate_sent_sms = raw_sent_sms.copy()
    intermediate_sent_sms.drop(
        columns=['created_at', 'updated_at', 'template_id', 'staff_id', 'sender', 'message_type'],
        inplace=True)

    intermediate_sent_sms.rename(
        columns={'message': 'body', 'date_sent': 'message_sent_date'},
        inplace=True)
    intermediate_sent_sms['body'].replace('', np.nan, inplace=True)
    intermediate_sent_sms.dropna(subset=['body'], inplace=True)

    return intermediate_sent_sms

