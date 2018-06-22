import os
import datetime as dt

import pandas as pd
from tenacity import retry, stop_after_attempt
from pandas_datareader import data as pdr
import fix_yahoo_finance as yf
from yahoo_fin.stock_info import get_data

START_DATE = '2000-01-01'
END_DATE = dt.datetime.now().strftime("%Y-%m-%d")
TICKERS = [
   'AAPL',
   'MSFT',
   'BIDU',
   'TRIP',
   'AMAG'
]
DATA_DIR = './csv/'
DATA_NAME = '/stock_data.csv'

if not os.path.exists(DATA_DIR):
    os.mkdir(DATA_DIR)

def _process_data(df):
    """Casts date column as pd.Datetime & creates weekday number
    column.
    """
    df['date'] = pd.to_datetime(df['date'])
    df['day_number'] = df['date'].dt.weekday
    
    return df

class finance_data():
    def __init__(self, start_date=None, end_date=None, tickers=None):
        """Uses Yahoo_finance fix to download stock data
        
        Keyword arguments:
        start_date -- date from whence stock data to be downloaded, str
        end_date -- max date for stock data download, str
        tickers -- list of tickers to be downloaded, list
        """
        self.start_date = start_date
        self.end_date = end_date
        self.tickers = tickers
        
        # Define tickers to download
        if self.tickers == None:
            self.tickers = TICKERS
            
        # Define start, default is January first 2000
        if self.start_date == None:
            self.start_date = START_DATE
        # Define end date, defualt is today's date  
        if self.end_date == None:
            self.end_date = END_DATE
        
    def get_fix_yahoo_data(self, store=True):
        """Gets Financial Stock data via Yahoo Finance for tickers defined in 
        finance_data, if none specified a default set will be downloaded.
        
        Keyword arguments:
        store -- if True the downloaded data is stored to csv, boolean
        
        Returns:
        df -- pd.DataFrame including stock data
        tickers = list of tickers
        """
        print('\nDownloading Stock Data\n')
        # download Stock Data
        yf.pdr_override() 
        df = pdr.get_data_yahoo(
            self.tickers, 
            start=self.start_date,
            end=self.end_date
        )
        
        # convert pd.Panel to pd.Frame
        if len(self.tickers) > 1:
            df = df.to_frame()
        elif len(self.tickers) == 1:
            df['ticker'] = self.tickers[0]
            
        df = df.reset_index()
        
        # Rename columns
        df.columns = [col.lower().replace(' ','') for col in df.columns]
        df = df.rename(columns={'minor':'ticker'})
        
        # Some tickers will not have any data, remove these from ticker list
        self.tickers = list(df.ticker.unique())
                    
        # Process dates
        df = _process_data(df)
        
        # Store data
        if store:
            df.to_csv(DATA_DIR + DATA_NAME, index=False)
     
        return df, self.tickers
    
    def get_yahoo_fin_data(self, store=True):
        """Gets Financial Stock data via Yahoo Finance for tickers defined in 
        finance_data, if none specified a default set will be downloaded. This
        function acts as a back to get_data().
        
        Keyword arguments:
        store -- if True the downloaded data is stored to csv, boolean
        
        Returns
        df -- pd.DataFrame including stock data
        tickers = list of tickers
        """
        df = pd.DataFrame([])
        

        for ticker in self.tickers:
            df_ticker = get_data(
                ticker, 
                start_date=self.start_date, 
                end_date=self.end_date
            )
            
            df_ticker = df_ticker.reset_index()
            df_ticker = _process_data(df_ticker)
            
            df = pd.concat([df, df_ticker])
            
        df = df.reset_index(drop=True)
        
        # Store data
        if store:
            df.to_csv(DATA_DIR + DATA_NAME, index=False)
        
        # Some tickers will not have any data, remove these from ticker list
        self.tickers = list(df.ticker.unique())
        
        return df, self.tickers
    
    @retry(stop=stop_after_attempt(7)) 
    def main(self):
        """As fix yahoo finance can be a little unstable main is used to try both
        fix_yahoo_finance and yahoo_fin in order to decrease the chances that
        code produces an error if no connection made. Retry decorator is used to
        improve robustness of function
        """
        try:
            print('Trying fix_yahoo_finance...')
            df, self.tickers = self.get_fix_yahoo_data()
        except:
            print('Trying yahoo_fin...')
            df, self.tickers = self.get_yahoo_fin_data()
            
        return df, self.tickers       