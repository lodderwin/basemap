import os
import datetime as dt
import logging

import pandas as pd
from tenacity import retry, stop_after_attempt
from pandas_datareader import data as pdr
import fix_yahoo_finance as yf
from yahoo_fin.stock_info import get_data

logging.basicConfig(level=logging.INFO)


class FinanceData():
    def __init__(self, start_date='2000-01-01', 
                 end_date=dt.datetime.now().strftime("%Y-%m-%d"), 
                 tickers=None, data_dir=None):
        """Uses Yahoo_finance fix to download stock data
        
        Args:
        start_date (str): date from whence stock data to be downloaded
        end_date (str): max date for stock data download
        tickers (list): list of tickers to be downloaded
        data_dir (str): directory where data should be saved
        
        Attributes:
        """
        for param in [start_date, end_date, tickers, data_dir]:
            if not param:
                raise ValueError('Keyword param has not been specified')
            else:
                self.start_date = start_date
                self.end_date = end_date
                self.tickers = tickers
                
        if not os.path.exists(data_dir):
            logging.info('{} does not exist, creating directory'.format(data_dir))
            os.mkdir(data_dir)
            
        self.data_dir = data_dir
            
    def get_fix_yahoo_data(self, store=True):
        """Gets Financial Stock data via Yahoo Finance for tickers defined in 
        finance_data, if none specified a default set will be downloaded.
        
        Args:
            store (bool): if True the downloaded data is stored to csv, boolean
        
        Attributes:
            df (pd.DataFrame): including stock data
            tickers (list): list of tickers
        """
        logging.info('downloading stock data')
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
        self.df = self._process_data(df)
        
        # Store data
        if store:
            self.df.to_csv(os.path.join(self.data_dir, 'historical_prices.csv'), 
               index=False)
         
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
            df_ticker = self._process_data(df_ticker)
            
            df = pd.concat([df, df_ticker])
            
        df = df.reset_index()
        
        # Rename columns
        df.columns = [col.lower().replace(' ','') for col in df.columns]
        df = df.rename(columns={'minor':'ticker'})
        
        # Some tickers will not have any data, remove these from ticker list
        self.tickers = list(df.ticker.unique())
                    
        # Process dates
        self.df = self._process_data(df)
        
        # Store data
        if store:
            self.df.to_csv(os.path.join(self.data_dir, 'historical_prices.csv'), 
               index=False)
    
    @retry(stop=stop_after_attempt(7)) 
    def get_fin_data(self):
        """As fix yahoo finance can be a little unstable main is used to try both
        fix_yahoo_finance and yahoo_fin in order to decrease the chances that
        code produces an error if no connection made. Retry decorator is used to
        improve robustness of function
        """
        try:
            logging.info('Trying fix_yahoo_finance...')
            self.get_fix_yahoo_data()
        except:
            logging.info('Trying yahoo_fin...')
            self.get_yahoo_fin_data()
                
    @staticmethod
    def _process_data(df):
        """Casts date column as pd.Datetime & creates weekday number
        column.
        """
        df['date'] = pd.to_datetime(df['date'])
        df['day_number'] = df['date'].dt.weekday
        
        return df

