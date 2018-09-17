import os
import datetime as dt
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from pandas.tseries.offsets import BDay
import numpy as np

import lstm_utils as utils
import lstm_model

matplotlib.style.use('seaborn-darkgrid')
figsize = (9,9*9/16)

today = dt.datetime.now().strftime("%Y-%m-%d")
now = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# If the folder with today's date doesnt exist then make it
plot_folder = './plots/'
plot_folder_train = './x_train_plots/'
if not os.path.exists(plot_folder):
    os.mkdir(plot_folder)
    
def add_predictions_to_plt(df, model, stock,days_ahead):
    df['date'] = df['date'] + BDay(5)
    for window in range(0,17):
        X, X_nmd = utils.gen_X(df, window=window)
        
        predictions_nmd = lstm_model.predict(model, X_nmd, days_ahead)
        predictions = (predictions_nmd + 1) * X[0][0]
        
        # Prep prediction data for plotting
        df_pred = df[df.window == df.window.max() - window][['date','ticker']][-5:].copy()
        df_pred = df_pred.reset_index(drop=True)
        df_pred = pd.concat([df_pred, pd.DataFrame(predictions[0], columns=['close'])], 
                        axis=1)
        
        # Plot predicted stock prices   
        plt.plot(df_pred.date, df_pred.close, color='b', linestyle='--', alpha=0.6)
        

def plot_latest_prediction(days_ahead,df, predictions, stock, growth, mse,
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
    add_predictions_to_plt(processed_df, model, stock,days_ahead)
    
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
    #plt.show()
    
    return plt

def plot_investment(investment_dev, ticker, params, margin, window_length):
    investment_sim = './investment_sim/'
    matplotlib.style.use('seaborn-darkgrid')
    # Create plot
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)#, figsize=figsize)
    ax.plot(investment_dev, label='Investment of '+ticker)
    plt.title('Investment over time of ' +ticker, size=16)
    plt.xlabel('Days', size=13)
    plt.ylabel('Investment', size=13)
    if len(params.keys())>6:
        plt.figtext(0.5, 0.01, 'date created: ' + now +' '+ str(params['node1'])+' '+str(params['node2']) +' '+str(params['node3']) +' '+ str(params['batch_size']) +' '+ str(margin)+ ' '+str(window_length)+' '+str(params['epochs']), 
                horizontalalignment='center' , size=10)
    elif len(params.keys())==6:
        plt.figtext(0.5, 0.01, 'date created: ' + now +' '+ str(params['node1'])+' '+str(params['node2']) +' '+ str(params['batch_size']) +' '+ str(margin)+ ' '+str(window_length)+' '+str(params['epochs']), 
                horizontalalignment='center' , size=10)
    plt.savefig(investment_sim + ticker +'_'+str(window_length)+ '_investment_development.png',dpi=400)
#    plt.show()
#    plt.close()
    
def plot_investment_train(investment_dev, ticker,params,margin,window_length):
    investment_sim = './investment_sim/'
    matplotlib.style.use('seaborn-darkgrid')
    # Create plot
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)#, figsize=figsize)
    ax.plot(investment_dev, label='Investment of '+ticker)
    plt.title('Investment over time of ' +ticker, size=16)
    plt.xlabel('Days', size=13)
    plt.ylabel('Investment', size=13)
    if len(params.keys())>6:
        plt.figtext(0.5, 0.01, 'date created: ' + now +' '+ str(params['node1'])+' '+str(params['node2']) +' '+str(params['node3']) +' '+ str(params['batch_size']) +' '+ str(margin)+ ' '+str(window_length)+' '+str(params['epochs']), 
                horizontalalignment='center' , size=10)
    elif len(params.keys())==6:
        plt.figtext(0.5, 0.01, 'date created: ' + now +' '+ str(params['node1'])+' '+str(params['node2']) +' '+ str(params['batch_size']) +' '+ str(margin)+ ' '+str(window_length)+' '+str(params['epochs']), 
                horizontalalignment='center' , size=10)
    plt.savefig(investment_sim + ticker +'_'+str(window_length)+ '_investment_development_train.png',dpi=400)

    
def plot_results(real_prices, corrected_predicted_test, days_ahead,ticker):
    # Use seaborn styling
    matplotlib.style.use('seaborn-darkgrid')
    # Create plot
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)#, figsize=figsize)
    ax.plot(real_prices, label='True Data')
    #padding is used to set the new predictions to an appropiate distance from 0 days
    for i, data in enumerate(corrected_predicted_test):
            padding = [None for p in list(range(int(((i) * days_ahead)/1.0)))]
            plt.plot(padding+data, label='Prediction', alpha=0.6)

    plt.title('Predictions for ' + ticker, size=16)
    plt.xlabel('Days', size=13)
    plt.ylabel('Stock Price', size=13)
    plt.figtext(0.5, 0.01, 'date created: ' + now, 
                horizontalalignment='center', size=10)
    plt.savefig(plot_folder + ticker + '_predictions.png',dpi=400)
    #plt.show()
    plt.close()
        
def plot_results_days_average(real_prices, corrected_predicted_test, days_average,ticker):
    # Use seaborn styling
    matplotlib.style.use('seaborn-darkgrid')
    # Create plot
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)#, figsize=figsize)
    ax.plot(real_prices, label='True Data')
    ax.plot(corrected_predicted_test, label = 'Predicted average')
    #padding is used to set the new predictions to an appropiate distance from 0 days
#    for i, data in enumerate(corrected_predicted_test):
#            padding = [None for p in list(range(int(((i) * days_ahead)/1.0)))]
#            plt.plot(padding+data, label='Prediction', alpha=0.6)

    plt.title('Predictions for ' + ticker, size=16)
    plt.xlabel('Days', size=13)
    plt.ylabel('Stock Price', size=13)
    plt.figtext(0.5, 0.01, 'date created: ' + now, 
                horizontalalignment='center', size=10)
    plt.savefig(plot_folder + ticker + '_predictions.png',dpi=400)
    #plt.show()
    plt.close()
def histogram(lst_predictions_trian,ticker):
    investment_sim = './investment_sim/'
    fig = plt.figure(figsize=figsize)
    plt.hist(lst_predictions_trian, bins=np.arange(-0.1,0.11,0.01))
    plt.ylabel('Probability')
    plt.title(ticker+str(len(lst_predictions_trian)))
    plt.savefig(investment_sim+ticker +'_'+ 'histogram.png',dpi=400)

    
def plot_mcr(df_selection,df_selection_title_positive, df_selection_title_negative, ticker,length_new_df_main):
    investment_sim = './investment_sim/'
    fig = plt.figure(figsize=figsize)
    plt.scatter(np.asarray(df_selection['closeratio_plot'].tolist()), np.asarray(df_selection['investmentratio_plot'].tolist()), s=np.asarray(df_selection['cirkelsize_plot'].tolist())*1000)
    line = np.polyfit(np.asarray(df_selection['closeratio_plot'].tolist()), np.asarray(df_selection['investmentratio_plot'].tolist()), 1, w=np.asarray(df_selection['cirkelsize_plot'].tolist())*1000)
    x = np.arange(-1,1.01,0.01)
    y1 = line[0]*x + line[1]
    y = x
    plt.plot(x,y)
    plt.plot(x,y1)
    plt.xlabel('close ratio')
    plt.ylabel('investment ratio')
    plt.title('Above : '+str(df_selection_title_positive['cirkelsize_plot'].sum())+'   Below : '+ str(df_selection_title_negative['cirkelsize_plot'].sum()) + '   Number of days : '+str(df_selection['consecutive'].sum()) +'  Total length : '+ str(length_new_df_main))
    plt.savefig(investment_sim+ticker +'_'+ 'selection_graph.png',dpi=400)

        
        
        
