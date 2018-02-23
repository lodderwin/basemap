import matplotlib.pyplot as plt
import pandas as pd
import yahoo_reader as yr
import lstm_functions as lf

# Define the instruments to download
tickers = ['AAPL','MSFT','^GSPC','BIDU','TRIP','AMAG','QCOM']
# Create Historic data
df = yr.finance_data(tickers=tickers).getData()
# Store data to csv
df.to_csv('./csv/store_data.csv')

#%%
# What is this?
seq_len = 5
# What are these for?
model_layers = [1,5,16,1]
# attempts at what?
attempts = 20

for stock in tickers:
    best_model = 'shit'
    profit = 0.0
    data = df[(df['ticker']==stock)]['close'].tolist()
    x_train, y_train, x_test, y_test, y_test_correction =  lf.create_windows(data,seq_len,True)
    model = lf.build_model(model_layers)
    for i in range(int(attempts)):
        model.fit(
                x_train,
                y_train,
                batch_size=128,
                nb_epoch=1,
                validation_split=0.05)
        days_ahead = 5
        predicted_test = lf.predict_test(days_ahead, x_test, seq_len, model)
        predictions_in_function = int(x_test.shape[0]/seq_len)
        corrected_predicted_test = lf.correct_predict_test(seq_len, predicted_test, y_test_correction)
        turnover, investments = lf.invest_sim(corrected_predicted_test, y_test_correction)
        if turnover>profit:
            best_model = model
            profit = turnover
    predicted_test = lf.predict_test(days_ahead, x_test, seq_len, best_model)
    corrected_predicted_test = lf.correct_predict_test(seq_len, predicted_test, y_test_correction)
    lf.plot_results(y_test_correction, corrected_predicted_test, days_ahead)
    turnover, investments = lf.invest_sim(corrected_predicted_test, y_test_correction)
            