import pandas as pd
from barefoot_winnie.d03_primary.restructure_intermediate_input import restructure_intermediate_input


def create_primary_fb_conversations(intermediate_fb_messages: pd.DataFrame,
                                    gap_between_conversations: int = 10):

    # remove adverts
    msg_no_ads = intermediate_fb_messages.loc[~intermediate_fb_messages.body.str.contains("clicked on an ad")].copy()

    # identify conversations
    primary_fb = restructure_intermediate_input(intermediate_input=msg_no_ads,
                                                gap_between_conversations=gap_between_conversations)

    return primary_fb
