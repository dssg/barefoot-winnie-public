from barefoot_winnie.d03_primary.restructure_intermediate_input import restructure_intermediate_input


def create_primary_sms(intermediate_received_sms, intermediate_sent_sms,
                       gap_between_conversations: int = 10):

    # preparing questions received via sms
    intermediate_received_sms.rename(
        columns={'sender': 'thread_id'},
        inplace=True)
    intermediate_received_sms['from'] = intermediate_received_sms['thread_id']

    # preparing answers sent via sms
    intermediate_sent_sms.rename(
        columns={'recipient': 'thread_id'},
        inplace=True)
    intermediate_sent_sms['from'] = 'Barefoot Lawyers- Uganda'

    # combining to get same structure as fb data.
    merged_sms = intermediate_received_sms.append(intermediate_sent_sms)
    merged_sms = merged_sms[['id','thread_id', 'body', 'from', 'message_sent_date']]
    merged_sms['thread_id'] = merged_sms['thread_id'].astype(str)

    # running the restructuring function
    primary_sms = restructure_intermediate_input(intermediate_input=merged_sms,
                                                 gap_between_conversations=gap_between_conversations)

    return primary_sms
