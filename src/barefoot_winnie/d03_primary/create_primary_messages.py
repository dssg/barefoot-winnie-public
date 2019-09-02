
def create_primary_messages(primary_sms, primary_fb_conversations):
    """ Combines the Facebook and SMS messages into one add a field
        Filters 'simple' questions. Used as the proxy for identifying 'inquiries'
    """

    primary_sms['channel'] = 'SMS'
    primary_fb_conversations['channel'] = 'Facebook'
    primary_messages = primary_sms.append(primary_fb_conversations)
    primary_messages['total_num_messages'] = (primary_messages
                                              .groupby(['channel', 'thread_id', 'num_of_conversations'])
                                              .num_of_messages
                                              .transform(max)
                                              )
    primary_messages['one_interaction'] = False
    primary_messages.loc[primary_messages['total_num_messages'] == 1, 'one_interaction'] = True

    return primary_messages
