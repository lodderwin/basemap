import os
import datetime as dt
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from pandas.tseries.offsets import BDay

import lstm_utils as utils
import lstm_model

matplotlib.style.use('seaborn-darkgrid')
figsize = (9,9*9/16)

today = dt.datetime.now().strftime("%Y-%m-%d")
now = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# If the folder with today's date doesnt exist then make it
plot_folder = './plots/{}/'.format(today)
if not os.path.exists(plot_folder):
    os.mkdir(plot_folder)
    
def add_predictions_to_plt(df, model, stock):
    df['date'] = df['date'] + BDay(5)
    for window in range(0,17):
        X, X_nmd = utils.gen_X(df, window=window)
        
        predictions_nmd = lstm_model.predict(model, X_nmd)
        predictions = (predictions_nmd + 1) * X[0][0]
        
        # Prep prediction data for plotting
        df_pred = df[df.window == df.window.max() - window][['date','ticker']][-5:].copy()
        df_pred = df_pred.reset_index(drop=True)
        df_pred = pd.concat([df_pred, pd.DataFrame(predictions[0], columns=['close'])], 
                        axis=1)
        
        # Plot predicted stock prices   
        plt.plot(df_pred.date, df_pred.close, color='b', linestyle='--', alpha=0.6)
        

def plot_latest_prediction(df, predictions, stock, growth, mse,
                           model, processed_df, days_of_history=20):
    """ Line plot of historical stock price and predicted stock price.
    Actuals are plotted with solid line and prediction continues as dotted
    line.
    """
    # Prep historic data for plotting 
    df_hist = df[['date','close']][-days_of_history:]
    # Prep prediction data for plotting
#    df_pred = pd.DataFrame(df_hist['date'][-5:].copy(), columns=['date'])
#    df_pred = df_pred.reset_index(drop=True)
#    df_pred = df_pred['date'] + BDay(5)
#    df_pred = pd.concat([df_pred, pd.DataFrame(predictions[0], columns=['close'])], 
#                        axis=1)
    # Add the last value in df_hist to df_pred so when plotted the lines join
#    df_pred = pd.concat([df_hist[-1:], df_pred])
    
    # Plot historic and predictions as line plot    
    plt.figure(figsize=(10,5))

    # Plot histotic stock prices
    plt.plot(df_hist.date, df_hist.close, color='b', label='actual')
    
    # Plot predicted stock prices   
    add_predictions_to_plt(processed_df, model, stock)
    
    # Configure 
    plt.title('Expected growth for {}: {}% (mse: {})'.format(stock, 
                                                             round(growth, 2),
                                                             round(mse, 4)),
                                                             size=16)
    plt.ylabel('Price', size=13)
    plt.figtext(0.5, 0.01, 'date created: {}'.format(now), 
                horizontalalignment='center', size=10)
    
#    plt.legend(fontsize=13)
    
    plt.savefig('{}{}_latest_prediction.png'.format(plot_folder, stock), 
                dpi=150, bbox_inches='tight')
    plt.show()
    
    return plt



        
        
        
