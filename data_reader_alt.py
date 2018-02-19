from pandas_datareader import data as pdr
import fix_yahoo_finance as yf
import pandas as pd
import datetime as dt

class finance_data():
    def __init__(self, start_date=None, end_date=None):
        """
        Uses Yahoo_finance fix to download stock data
        """
        self.tickers = ['AAPL','MSFT','^GSPC',
                        'BIDU','TRIP','AMAG','QCOM']
        if start_date == None:
            self.start_date = '2000-01-01'
            
        if end_date == None:
            self.end_date = dt.datetime.now().strftime("%Y-%m-%d")
        
    def getData(self):
        """
        Gets Financial Stock data via Yahoo Finance
        
        returns
        --------
        df : pd.DataFrame including stock data
        """
        print('\nDownloading Stock Data\n')
        # download Stock Data
        yf.pdr_override() 
        df = pdr.get_data_yahoo(self.tickers, 
                                start=self.start_date, 
                                end=self.end_date)
        # convert pd.Panel to pd.Frame
        df = df.to_frame()
        df = df.reset_index()
        # Rename columns
        df.columns = [col.lower() for col in df.columns]
        df = df.rename(columns={'minor':'ticker'})
        
        return df

#%%
fd = finance_data()
df = fd.getData()

df.head()
