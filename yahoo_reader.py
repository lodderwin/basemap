from pandas_datareader import data as pdr
import fix_yahoo_finance as yf
import pandas as pd
import datetime as dt

class finance_data():
    def __init__(self, start_date=None, end_date=None, tickers=None):
        """
        Uses Yahoo_finance fix to download stock data
        
        Parameters
        start_date : date from whence stock data to be downloaded, str
        end_date : max date for stock data download, str
        tickers : list of tickers to be downloaded, list
        """
        self.start_date = start_date
        self.end_date = end_date
        self.tickers = tickers
        
        # Define tickers to download
        if self.tickers == None:
            self.tickers = ['AAPL','MSFT','^GSPC',
                            'BIDU','TRIP','AMAG',
                            'QCOM']
            
        # Define start, default is January first 2000
        if self.start_date == None:
            self.start_date = '2000-01-01'
        # Define end date, defualt is today's date  
        if self.end_date == None:
            self.end_date = dt.datetime.now().strftime("%Y-%m-%d")
        
    def get_data(self, store=True):
        """
        Gets Financial Stock data via Yahoo Finance for tickers defined in 
        finance_data, if none specified a default set will be downloaded.
        
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
        if len(self.tickers) > 1:
            df = df.to_frame()
        df = df.reset_index()
        # Rename columns
        df.columns = [col.lower() for col in df.columns]
        df = df.rename(columns={'minor':'ticker'})
        # Store data
        if store:
            df.to_csv('./csv/stock_data.csv', index=False)
        
        return df
