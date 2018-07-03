import yaml

import pandas as pd
from slackclient import SlackClient

TOKEN = yaml.load(open('./configs/slack_token.yml'))['token']
SC = SlackClient(TOKEN)

def _gen_prediction_ranking_text(df : pd.DataFrame):
    msg = ":loudspeaker: *Latest Predictions*\nTop Tickers _ranked by predicted growth_\n"
    for i, row in df[10:].iterrows():
        msg = msg + ("{}. *{}: {}* _({})_\n"
            .format(row.ranking, row.ticker, row.growth, row.margin)
        )
    
    return msg

def _prep_predictions_for_message(df : pd.DataFrame):
    df = (df
       .query('growth_mt_margin == 1')
       .query('growth > 1')
       .reset_index(drop=True)
       .assign(ranking=lambda x: x.index + 1)
    )
    
    return df

class PyntBot():    
    def __init__(self, df=None):
        if df is not None:
            self.df = _prep_predictions_for_message(df)
            self.message = _gen_prediction_ranking_text(self.df)
        
    def send_top_tickers(self):
        SC.api_call(
            'chat.postMessage',
             channel='slackbot_test',
             text=self.message,
             username='pyntbot',
             footer='pyntbot'
        )