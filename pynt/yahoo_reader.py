import os
import datetime as dt
import logging

import pandas as pd
from tenacity import retry, stop_after_attempt, wait_fixed
from pandas_datareader import data as pdr
import fix_yahoo_finance as yf
from yahoo_fin.stock_info import get_data

logging.basicConfig(level=logging.INFO)


class FinanceData():
    def __init__(self, start_date='2000-01-01',
                 end_date=dt.datetime.now().strftime("%Y-%m-%d"),
                 tickers=None, data_dir=None, window_length=6):
        """Uses Yahoo_finance fix to download stock data
        
        Args:
        start_date (str): date from whence stock data to be downloaded
        end_date (str): max date for stock data download
        tickers (list): list of tickers to be downloaded
        data_dir (str): directory where data should be saved
        
        Attributes:
        start_date (str): date from whence stock data to be downloaded
        end_date (str): max date for stock data download
        tickers (list): list of tickers to be downloaded
        data_dir (str): directory where data should be saved
        """
        self.start_date = start_date
        self.end_date = end_date
        self.tickers = tickers
        self.window_length = window_length

        if not os.path.exists(data_dir):
            logging.info('{} does not exist, creating directory'.format(data_dir))
            os.mkdir(data_dir)

        self.data_dir = data_dir

        self.df = self.get_yahoo_data(store=False)
        self.df_processed = self.pre_process_data()

    @retry(stop=stop_after_attempt(10), wait=wait_fixed(10))
    def get_yahoo_data(self, store=True):
        """Gets Financial Stock data via Yahoo Finance for tickers defined in 
        finance_data, if none specified a default set will be downloaded.
        
        Args:
            store (bool): if True the downloaded data is stored to csv, boolean
        
        Attributes:
            df (pd.DataFrame): including stock data
            tickers (list): list of tickers
        """
        try:
            logging.info('downloading stock data')
            # download data using fix yahoo package
            try:
                logging.info('trying fix yahoo package')
                yf.pdr_override()
                df = pdr.get_data_yahoo(
                    self.tickers,
                    start=self.start_date,
                    end=self.end_date
                )
            except Exception:
                try:
                    logging.error('fix yahoo package failed')
                    logging.info('trying yahoo fin package')
                    # try using yahoo_fin package
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
                except Exception:
                    logging.error('could not download yahoo data')
                    raise

            if len(df) == 0:
                raise ValueError('no data downloaded')

            # convert pd.Panel to pd.Frame
            if len(self.tickers) > 1:
                df = df.to_frame()
            elif len(self.tickers) == 1:
                df['ticker'] = self.tickers[0]

            df = df.reset_index()

            df.columns = [col.lower().replace(' ', '') for col in df.columns]
            df = df.rename(columns={'minor': 'ticker'})

            self.tickers = list(df.ticker.unique())

            logging.info('{} rows downloaded for {} tickers'.
                         format(len(df), len(self.tickers)))

            df['date'] = pd.to_datetime(df['date'])
            df['day_number'] = df['date'].dt.weekday

            # Store data
            if store:
                df.to_csv(os.path.join(self.data_dir, 'historical_prices.csv'), index=False)

            return df

        except Exception:
            logging.error("cannot connect to yahoo server")
            raise

    def pre_process_data(self):
        """
        Processes data for LSTM model

        Parameters
        --------
        df : pd.DataFrame with cols: ticker, date, close

        Returns
        --------
        df : pd.DataFrame
        """
        self.df['date'] = pd.to_datetime(self.df.date)
        self.df = self.df.sort_values(['ticker', 'date'])
        # Create empty df to be filled with windows normalised for each ticker
        df_processed = pd.DataFrame([])
        tickers = list(self.df.ticker.unique())
        # For each ticker in df create normalised Windows
        for idx, ticker in enumerate(tickers):
            logging.info('creating normalised windows for {} ({}/{})'
                         .format(ticker, idx + 1, len(tickers)))
            df_temp = self.df[self.df.ticker == ticker]
            df_temp = self.normalise_windows(df_temp, window_length=self.window_length)
            df_processed = pd.concat([df_processed, df_temp])

        return df_processed.reset_index(drop=True)

    @staticmethod
    def normalise_windows(df, window_length=6):
        """
        Takes a DataFrame as input and returns a much larger DataFrame with
        normalised windows. A window of data is a run of dates of size window
        length. The number of windows created is dependant on the length of
        the DataFrame and the window size: Windows = Length - Window size.
        For each window the data is normalised by dividing by the first row of
        each window.

        Parameters
        --------
        df : pd.DataFrame incl columns: ticker, date, close
        window_length : size of Window = train + test

        Returns
        --------
        df : df with extra columns: window (window id), normaliser (for
        de-normalisation), close_nmd (nomrlaised close price)
        """
        # Minus 5 instead of 6 due to range function used in loop
        windows = len(df) - (window_length - 1)
        # Create empty dataframe to be filled with windows
        df_final = pd.DataFrame([])

        for i in range(0, windows):
            # Print progress counter
            logging.info('normalising window {} of {}'.format(i + 1, windows))
            # Create a dataframe for every 6 rows
            df_temp = df[i:i + window_length]
            df_temp = df_temp.reset_index(drop=True)
            df_temp['window'] = i + 1
            df_temp['normaliser'] = df_temp.close[0]
            df_temp['normaliserv'] = df_temp.volume[0]
            df_temp['close_nmd'] = (df_temp.close / df_temp.normaliser) - 1
            df_temp['volume_nmd'] = (df_temp.volume / df_temp.normaliserv) - 1
            df_temp['normaliserh'] = df_temp.high[0]
            df_temp['normaliserl'] = df_temp.low[0]

            df_final = pd.concat([df_final, df_temp]).reset_index(drop=True)

        return df_final
