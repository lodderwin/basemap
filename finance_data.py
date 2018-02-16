#%%
import quandl
import pandas as pd
import yaml

class finance_data:
    def __init__(self, api_yml=None):
        """
        Intialises finance_data class returning api_key, a dataframe including all
        company codes listed in EOD, all company codes and company names. 
        """
        # Retrieve api key
        if api_yml == None:
            api_yml = './api.yml'
        # Load Personal api key
        self.api_key = yaml.load(open(api_yml))['key']
        # Confgiure quandl api
        quandl.ApiConfig.api_key = self.api_key
        quandl.ApiConfig.api_version = '2015-04-09'
        
        # Load EOD data
        # Read quandl stock codes
        # csv downloaded from https://www.quandl.com/api/v3/databases/wiki/codes
        df = pd.read_csv('./csv/EOD-datasets-codes.csv', 
                         header=None, 
                         names=['code','comp_name'])
        # Sort alphabetically & reset index
        df = df.sort_values('code')
        self.df_codes = df.reset_index(drop=True)
        # Get codes and names as list
#        self.codes = self.df_codes.code.tolist()
        # Codes below are available for non-premium users
        self.codes = ['EOD/AAPL','EOD/AXP','EOD/BA','EOD/CAT','EOD/CSCO','EOD/CVX',
                      'EOD/DIS','EOD/GE','EOD/GS','EOD/HD','EOD/IBM','EOD/INTC',
                      'EOD/JNJ','EOD/JPM','EOD/KO','EOD/MCD','EOD/MMM','EOD/MRK',
                      'EOD/MSFT','EOD/NKE','EOD/PFE','EOD/PG','EOD/TRV','EOD/UNH',
                      'EOD/UTX','EOD/V','EOD/VZ','EOD/WMT','EOD/XOM']
        self.names = self.df_codes.comp_name.tolist()
        
    def getData(self):
        """
        Gets financial stock data from qunadl.
        
        Parameters
        --------
        self
        
        Return
        --------
        data : DataFrame with all financial info for all companies listed in
        self.code
        """
        # Empty df to be filled with financial data during loop below
        df = pd.DataFrame([])
        for idx, code in enumerate(self.codes):
            # Not all data is available without premium subscription
            try:
                print('Getting data for', idx + 1,'of',len(self.codes))
                # get data from quandl
                df_temp = quandl.get(code)
                # New columns: 1. code & 2. company name
                df_temp['code'] = code[-4:].replace('/','')
                df_temp['source'] = code[:3]
                df_temp['name'] = self.names[idx]
                # Reset Index
                df_temp = df_temp.reset_index()
                # column headers in lower case
                df_temp.columns = [col.lower() for col in df_temp.columns]
                # concate to df
                df = pd.concat([df, df_temp])
            except:
                pass
        
        return df