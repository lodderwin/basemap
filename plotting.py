import os
import datetime as dt
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from pandas.tseries.offsets import BDay

matplotlib.style.use('seaborn-darkgrid')
figsize = (9,9*9/16)

today = dt.datetime.now().strftime("%Y-%m-%d")
now = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# If the folder with today's date doesnt exist then make it
plot_folder = './plots/' + today + '/'
if not os.path.exists(plot_folder):
    os.mkdir(plot_folder)

def plot_latest_prediction(df, predictions, stock, growth, days_of_history=20):
    """ Line plot of historical stock price and predicted stock price.
    Actuals are plotted with solid line and prediction continues as dotted
    line.
    """
    # Prep historic data for plotting
    
    df_hist = df[['date','close']][-days_of_history:]
    # Prep prediction data for plotting
    df_pred = pd.DataFrame(df_hist['date'][-5:].copy(), columns=['date'])
    df_pred = df_pred.reset_index(drop=True)
    df_pred = df_pred['date'] + BDay(5)
    df_pred = pd.concat([df_pred, pd.DataFrame(predictions[0], columns=['close'])], 
                        axis=1)
    # Add the last value in df_hist to df_pred so when plotted the lines join
    df_pred = pd.concat([df_hist[-1:], df_pred])
    
    # Plot historic and predictions as line plot    
    plt.figure(figsize=(10,5))

    # Plot histotic stock prices
    plt.plot(df_hist.date, df_hist.close, color='b', label='actual')
    
    # Plot predicted stock prices   
    plt.plot(df_pred.date, df_pred.close, color='b', linestyle='--', label='predicted')
    
    # Configure 
    plt.title('Expected growth for ' + stock + ': ' + str(growth) + '%',
              size=16)
    plt.ylabel('Price', size=13)
    plt.figtext(0.5, 0.01, 'date created: ' + now, 
                horizontalalignment='center', size=10)
    
    plt.legend(fontsize=13)
    
    plt.savefig(plot_folder + 'latest_prediction_' + stock + '.png', 
                dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()
