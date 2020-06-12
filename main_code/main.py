import warnings
warnings.filterwarnings("ignore")
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import svm
from sklearn import preprocessing as pre
import pickle
from Portfolio import Portfolio
from Fuzzy_logic import Fuzzy_logic
#import plotly.plotly as py
from matplotlib.finance import candlestick2_ohlc
import lstm
#import plotly.graph_objs as go
from keras.models import load_model
import h5py
import os


def main():

    # # this code is for loading the saved figure:
    # with open('./saved_files/plots/'+'Budgets_2.pickle', 'rb') as fid:
    #     fig, ax = pickle.load(fid)
    # plt.show(fig)
    # input('hi')

    # ------ Settings:
    path_dataset = './dataset/prices-split-adjusted.csv'
    path_dataset_fundamental = './dataset/fundamentals.csv'
    path_save = './saved_files/'    # GE, GOOG, ORCL and JNJ have no fundamental features, GM has not enough technical samples
    # 'AAPL', 'AIG', 'AMZN', 'BA', 'CAT', 'COF', 'EBAY', 'F', 'FDX', 'GE', 'GM', 'GOOG', 'HD', 'IBM', 'JNJ', 'JPM', 'KO', \
    # 'MSFT', 'NFLX', 'NKE', 'ORCL', 'PEP', 'T', 'WMT', 'XOM', 'XRX'
    stocks_to_process = ['AAPL', 'AIG', 'AMZN', 'BA', 'CAT', 'COF', 'EBAY', 'F', 'FDX', 'HD', 'IBM', 'JPM', 'KO', 'MSFT', 'NKE', 'PEP', 'T', 'WMT', 'XOM', 'XRX']
    # , 'BA', 'CAT', 'COF', 'EBAY', 'F', 'FDX', 'HD', 'IBM', 'JPM', 'KO', 'MSFT', 'NFLX', 'NKE', 'PEP', 'T', 'WMT', 'XOM', 'XRX']#'AAPL', 'AIG', 'AMZN', 'BA', 'CAT', 'COF', 'EBAY', 'F', 'FDX', 'HD', 'IBM', 'JPM', 'KO', 'MSFT', 'NKE', 'PEP', 'T', 'WMT', 'XOM', 'XRX']
    # , 'BA', 'CAT', 'COF', 'EBAY', 'F', 'FDX', 'HD', 'IBM', 'JPM', 'KO', 'MSFT', 'NFLX', 'NKE', 'PEP', 'T', 'WMT', 'XOM', 'XRX'

    # 'GE', 'CAT','GM', 'IBM', 'AAPL', 'AMZN', 'NFLX', 'F'
    # stocks_to_process = ['AAPL', 'AIG', 'AMZN']
    number_of_stocks = len(stocks_to_process)
    fundamental_features = ['Accounts Receivable', 'Capital Expenditures', 'Inventory', 'Gross Margin', 'Income Tax']
    include_technical_analysis = True
    include_fundamental_analysis = True
    do_plot = True
    fraction_of_training_set = 0.8
    predict_which_stock_variable = 3   # 3: highest
    train_again = True
    do_suggest_weights = True
    n_smoothing_period = 14

    optimum_parameters_in_stocks = [5, 1000, 0.001]   # order of parameters: window_size, C, gamma [15, 1000, 0.0001]

    apply_moving_average = True
    movingAverage_windowSize = 20

    initial_day_in_testSet = 150
    target_day_range = 30
    window_size_covariance_calculation = 100

    risk_tolerance_percentage = 30
    predictor_type = 'SVR'   # can be either 'SVR' or 'LSTM'
    LSTM_epochs = 50
    Initial_Budget = 1000  # 1000 $

    # ------ loading dataset:
    data_df = pd.read_csv(path_dataset, delimiter=',')
    data = data_df.values  # converting pandas data frame to numpy array
    actual_testSet_AllStocks_targetStockVariable_percentage = []
    predicted_testSet_AllStocks_targetStockVariable_percentage = []
    actual_AllDataset_AllStocks_targetStockVariable_percentage = []
    actual_AllDataset_AllStocks_targetStockVariable_percentage_averaged = []
    actual_AllDatasetset_stockVariable_list = []
    actual_testingSet_AllStocks_targetStockVariable = []

    # ------ Preprocessing & time series forcasting:
    print('Status of code: Preprocessing & time series forcasting.....')
    for stock_index in range(len(stocks_to_process)):
        name_of_stock = stocks_to_process[stock_index]

        # ---- Settings of SVR training
        window_size_of_observation_for_prediction = optimum_parameters_in_stocks[0]
        c = optimum_parameters_in_stocks[1]
        gamma = optimum_parameters_in_stocks[2]

        data_a_stock = read_data_of_a_stock(dataset=data, name_of_stock=name_of_stock)

        # ------ pre-processing of dataset:
        data_a_stock = extract_features_from_dataset(dataset=data_a_stock)
        total_number_of_days = data_a_stock.shape[0]

        closing = np.reshape(data_a_stock[:, 1], [-1, 1])
        # ------ RSI:


        # changes_closing = np.zeros((total_number_of_days, 1))
        # changes_closing[1:] = closing[1:] - closing[0:-1]
        # Gain_closing = np.maximum(0, changes_closing)
        # Loss_closing = np.abs(np.minimum(0, changes_closing))
        # AG, AL = np.zeros((total_number_of_days, 1)), np.zeros((total_number_of_days, 1))
        # AG[n_smoothing_period] = np.mean(Gain_closing[0:n_smoothing_period]) # np.max([np.mean(Gain_closing[0:n_smoothing_period]), 0.00001])
        # AL[n_smoothing_period] = np.mean(Loss_closing[0:n_smoothing_period]) # np.max([np.mean(Loss_closing[0:n_smoothing_period]), 0.00001])
        # for day_index in range(n_smoothing_period+1, total_number_of_days):
        #     AG[day_index] = (AG[day_index - 1] * (n_smoothing_period - 1) + Gain_closing[day_index]) / n_smoothing_period
        #     AL[day_index] = (AL[day_index - 1] * (n_smoothing_period - 1) + Loss_closing[day_index]) / n_smoothing_period
        # RS = AG / AL
        # RSI = 100 - (100 / (1 + RS))

        # ------ ADX:
        opening = np.reshape(data_a_stock[:, 0], [-1, 1])
        lowest = np.reshape(data_a_stock[:, 2], [-1, 1])
        highest = np.reshape(data_a_stock[:, 3], [-1, 1])
        yesterday_closing = np.zeros((total_number_of_days, 1))
        yesterday_highest = np.zeros((total_number_of_days, 1))
        yesterday_lowest = np.zeros((total_number_of_days, 1))
        yesterday_closing[1:] = closing[0:-1]
        yesterday_highest[1:] = highest[0:-1]
        yesterday_lowest[1:] = lowest[0:-1]
        TR = np.maximum(highest - lowest, np.abs(highest - yesterday_closing), np.abs(lowest - yesterday_closing))
        STR = np.zeros((total_number_of_days, 1))
        STR[n_smoothing_period] = np.mean(TR[0:n_smoothing_period]) # np.max([np.mean(TR[0:n_smoothing_period]), 0.00001])
        PDM = np.maximum(highest - yesterday_highest, 0)
        SPDM = np.zeros((total_number_of_days, 1))
        SPDM[n_smoothing_period] = np.mean(PDM[0:n_smoothing_period]) # np.max([np.mean(PDM[0:n_smoothing_period]), 0.00001])
        NDM = np.maximum(yesterday_lowest - lowest, 0)
        SNDM = np.zeros((total_number_of_days, 1))
        SNDM[n_smoothing_period] = np.mean(NDM[0:n_smoothing_period]) # np.max([np.mean(NDM[0:n_smoothing_period]), 0.00001])
        for day_index in range(n_smoothing_period + 1, total_number_of_days):
            STR[day_index] = (STR[day_index - 1] * (n_smoothing_period - 1) + TR[day_index]) / n_smoothing_period
            SPDM[day_index] = (SPDM[day_index - 1] * (n_smoothing_period - 1) + PDM[day_index]) / n_smoothing_period
            SNDM[day_index] = (SNDM[day_index - 1] * (n_smoothing_period - 1) + NDM[day_index]) / n_smoothing_period
        SPDI = 100 * SPDM / STR
        SNDI = 100 * SNDM / STR
        DX = 100* np.abs(SPDI - SNDI) / (SPDI + SNDI)
        ADX = np.zeros((total_number_of_days, 1))
        ADX[2*n_smoothing_period] = np.max([np.mean(DX[n_smoothing_period:2*n_smoothing_period]), 0.00001])
        for day_index in range(n_smoothing_period + 1, total_number_of_days):
            ADX[day_index] = (ADX[day_index - 1] * (n_smoothing_period - 1) + DX[day_index]) / n_smoothing_period

        # ------ SAR:
        SAR = np.zeros((total_number_of_days, 1))
        SAR[3] = np.min(lowest[0:4])  # assume uptrend
        trend = 'uptrend'
        if SAR[3] > lowest[3]:
            # is downtrend
            trend = 'downtrend'
            SAR[3] = np.max(highest[0:4])
        AF = np.zeros((total_number_of_days, 1))
        AF[3] = 0.02
        EP = np.zeros((total_number_of_days, 1))
        if SAR[3] < highest[3]:
            # is uptrend
            EP[3] = np.max(highest[0:4])
        else:
            # is downtrend
            EP[3] = np.min(lowest[0:4])
        for day_index in range(3, total_number_of_days-1):
            trend_changes = False
            if trend == 'uptrend':
                # assume uptrend
                if day_index != 3:
                    EP[day_index] = np.max([EP[day_index-1], highest[day_index]])
                if SAR[day_index] > lowest[day_index]:
                    # is downtrend
                    trend_changes = True
                    trend = 'downtrend'
                    EP[day_index] = np.min(lowest[day_index-3:day_index+1])
                    AF[day_index] = 0.02
                    SAR[day_index+1] = np.max(highest[day_index-3:day_index+1])
            elif trend == 'downtrend':
                # assume downtrend
                EP[day_index] = np.min([EP[day_index - 1], lowest[day_index]])
                if SAR[day_index] < highest[day_index]:
                    # is uptrend
                    trend_changes = True
                    trend = 'uptrend'
                    EP[day_index] = np.max(highest[day_index-3:day_index+1])
                    AF[day_index] = 0.02
                    SAR[day_index + 1] = np.min(lowest[day_index - 3:day_index + 1])
            if trend_changes == False and day_index != 3:
                if EP[day_index] != EP[day_index-1]:
                    AF[day_index] = np.min([AF[day_index-1] + 0.02, 0.2])
                else:
                    AF[day_index] = AF[day_index - 1]
            if trend_changes == False:
                SAR[day_index+1] = SAR[day_index] + AF[day_index]*(EP[day_index]-SAR[day_index])
        # ------ adding indices to the data:
        # data_a_stock = np.delete(data_a_stock, 0, 1)  # delete opening
        # data_a_stock = np.delete(data_a_stock, 0, 1)  # delete closing
        # data_a_stock = np.delete(data_a_stock, 0, 1)  # delete lowest
        # data_a_stock = np.delete(data_a_stock, 0, 1)  # delete highest
        # data_a_stock = np.delete(data_a_stock, 4, 1)  # delete volume
        # data_a_stock = np.column_stack((data_a_stock, RSI))
        data_a_stock = np.column_stack((data_a_stock, ADX))
        data_a_stock = np.column_stack((data_a_stock, SAR))
        data_a_stock = data_a_stock[2*n_smoothing_period:]
        # ------
        if stock_index == 0:
            number_of_stock_variables = data_a_stock.shape[1]
            hit_rate_all_stocks = np.empty((0, number_of_stock_variables))
            MAE_all_stocks = np.empty((0, number_of_stock_variables))
            RMSE_all_stocks = np.empty((0, number_of_stock_variables))

        # ------ splitting data into training and testing:
        training_set_withValidationSet_not_averaged, testing_set_not_averaged = divide_dataset_to_training_and_testing_sets(
            dataset=data_a_stock, fraction_of_training_set=fraction_of_training_set)
        training_set_not_averaged, _ = divide_dataset_to_training_and_testing_sets(
            dataset=training_set_withValidationSet_not_averaged,
            fraction_of_training_set=0.8)  # here, second output is validation set
        if apply_moving_average:
            data_a_stock_averaged = np.zeros(data_a_stock.shape)
            # ------ moving average:
            # take Openning, Closing, Low, High, Volume:
            OpenningClosingLowHighVolume = data_a_stock.copy()
            OpenningClosingLowHighVolume = OpenningClosingLowHighVolume[:, :5]

            # pad at the first not to change length of time series after moving average:
            OpenningClosingLowHighVolume_firstDay = OpenningClosingLowHighVolume[0, :]
            OpenningClosingLowHighVolume_firstDay_repeat = np.tile(OpenningClosingLowHighVolume_firstDay, (movingAverage_windowSize - 1, 1))
            OpenningClosingLowHighVolume_padded = np.vstack((OpenningClosingLowHighVolume_firstDay_repeat, OpenningClosingLowHighVolume))
            # apply moving average:
            for column_index in range(OpenningClosingLowHighVolume_padded.shape[1]):
                time_series = OpenningClosingLowHighVolume_padded[:, column_index]
                time_series_MovingAveraged = moving_average(data_set=time_series.ravel(), periods=movingAverage_windowSize)
                OpenningClosingLowHighVolume[:, column_index] = time_series_MovingAveraged
            # overwrite Openning, Closing, Low, High, Volume with their moving average:
            data_a_stock_averaged[:, :5] = OpenningClosingLowHighVolume
            data_a_stock_averaged[:, 5:] = data_a_stock[:, 5:]

            # ------ splitting data into training and testing:
            training_set_withValidationSet, testing_set = divide_dataset_to_training_and_testing_sets(dataset=data_a_stock_averaged, fraction_of_training_set=fraction_of_training_set)
            training_set, _ = divide_dataset_to_training_and_testing_sets(dataset=training_set_withValidationSet,
                                                                                    fraction_of_training_set=0.8)  # here, second output is validation set
            # ------ removing the first row of actual test set and the last row of predicted test set: --> because we are predicting tomorrow
            testing_set_not_averaged = testing_set_not_averaged[1:, :]

        else:
            training_set_withValidationSet = training_set_withValidationSet_not_averaged
            testing_set = testing_set_not_averaged
            training_set = training_set_not_averaged

        # ------ normalization:
        scaler_training = pre.StandardScaler().fit(X=training_set)
        training_set_scaled = scaler_training.transform(X=training_set)

        number_of_stock_variables = training_set_scaled.shape[1]

        # ------ preparing train and test data:
        X_train = training_set_scaled[:, :]
        X_test = testing_set[:, :]
        Z_test = np.zeros(testing_set.shape)
        for window_index in range(1, window_size_of_observation_for_prediction):
            Z_train = np.zeros(training_set_scaled.shape)
            Z_train[window_index:, :] = training_set_scaled[:-window_index, :]
            X_train = np.column_stack((X_train, Z_train))
            Z_test[window_index:, :] = testing_set[:-window_index, :]
            Z_test[:window_index, :] = training_set_withValidationSet[-window_index:, :]
            X_test = np.column_stack((X_test, Z_test))
        X_train = np.delete(X_train, range(0, window_size_of_observation_for_prediction-1), axis=0)   # removing first non-valid rows
        X_train = X_train[:-1, :]      # today (last day does not have tomorrow and should be removed)

        # ------ Scaling test set:
        Number_of_test_samples = X_test.shape[0]
        scaler_testing_list = []
        for test_sample_index in range(Number_of_test_samples):
            stacked_previous_days_for_a_day = np.zeros((window_size_of_observation_for_prediction, number_of_stock_variables))
            for day_index in range(window_size_of_observation_for_prediction):
                stacked_previous_days_for_a_day[day_index, :] = X_test[test_sample_index, day_index*number_of_stock_variables:(day_index+1)*number_of_stock_variables]
            scaler_testing = pre.StandardScaler().fit(X=stacked_previous_days_for_a_day)
            scaled_stacked_previous_days_for_a_day = scaler_testing.transform(X=stacked_previous_days_for_a_day)
            X_test_a_sample = np.array([])
            for day_index in range(scaled_stacked_previous_days_for_a_day.shape[0]):
                stock_variables_in_day = (scaled_stacked_previous_days_for_a_day[day_index, :])
                X_test_a_sample = np.hstack((X_test_a_sample, stock_variables_in_day))
            X_test[test_sample_index, :] = X_test_a_sample
            scaler_testing_list.append(scaler_testing)

        # ------ training and testing the time series predictor:
        SVR_model_list = []
        LSTM_model_list = []
        predicted_stock = np.ndarray(shape=(testing_set.shape[0], 0))
        for stock_variable_index in range(number_of_stock_variables): # iterating over: Openning, Closing, Low, High, Volume, RSI, ADX, SAR
            # ------ time series prediction - training:
            y_train = training_set_scaled[window_size_of_observation_for_prediction:, stock_variable_index] # tomorrow (because we have cut the first window_size-1 rows of X, so we should remove window_size-1 first rows of y too)
            if train_again:
                if predictor_type == 'SVR':
                    SVR_model = svm.SVR(kernel='rbf', C=c, gamma=gamma).fit(X=X_train, y=y_train)
                    # SVR_model = svm.SVR(kernel='linear', C=c, gamma=gamma).fit(X=X_train, y=y_train)
                    SVR_model_list.append(SVR_model)
                    name_to_save = 'SVR_model_list_' + name_of_stock + '_StockVar' + str(stock_variable_index) + '_Window' + str(window_size_of_observation_for_prediction)
                    save_data(data=SVR_model_list, name_to_save=name_to_save, path=path_save+name_of_stock+'/')
                elif predictor_type == 'LSTM':
                    LSTM_model = lstm.build_model(layers=[1, X_train.shape[1], 100, 1])
                    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
                    results_train_LSTM = LSTM_model.fit(X_train, y_train, batch_size=10, nb_epoch=LSTM_epochs)
                    LSTM_model_list.append(LSTM_model)
                    # ------ save the LSTM model:
                    # https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model
                    # https://github.com/keras-team/keras/issues/1069
                    # google it: python how to save Sequential keras trained files
                    # https://machinelearningmastery.com/save-load-keras-deep-learning-models/
                    name_of_file_to_save = 'LSTM_model_' + name_of_stock + '_StockVar' + str(stock_variable_index)  + '_Window' + str(window_size_of_observation_for_prediction) +  '.h5'
                    LSTM_model.save_weights(path_save + name_of_file_to_save)
                    # ------ plot the history (loss) curve of training LSTM Keras:
                    # https://machinelearningmastery.com/display-deep-learning-model-training-history-in-keras/
                    # https://groups.google.com/forum/#!topic/keras-users/IVAarPXhv9A
                    history = results_train_LSTM.history
                    name_to_save = 'LSTM_loss_history_' + name_of_stock + '_StockVar' + str(stock_variable_index)  + '_Window' + str(window_size_of_observation_for_prediction)
                    save_data(data=history, name_to_save=name_to_save, path=path_save)
            else:
                if predictor_type == 'SVR':
                    name_to_load = 'SVR_model_list_' + name_of_stock + '_StockVar' + str(stock_variable_index)  + '_Window' + str(window_size_of_observation_for_prediction)
                    SVR_model_list = load_data(name_to_load=name_to_load, path=path_save+name_of_stock+'/')
                elif predictor_type == 'LSTM':
                    LSTM_model = lstm.build_model([1, X_train.shape[1], 100, 1])
                    LSTM_model_list = []
                    for stock_var_index_to_load in range(number_of_stock_variables):
                        name_of_file_to_load = 'LSTM_model_' + name_of_stock + '_StockVar' + str(stock_variable_index)  + '_Window' + str(window_size_of_observation_for_prediction) + '.h5'
                        LSTM_model.load_weights(path_save + name_of_file_to_load)
                        LSTM_model_list.append(LSTM_model)

            # ------ plot the history (loss) curve of training LSTM Keras:
            if predictor_type == 'LSTM':
                # https://machinelearningmastery.com/display-deep-learning-model-training-history-in-keras/
                # https://groups.google.com/forum/#!topic/keras-users/IVAarPXhv9A
                name_to_save = 'LSTM_loss_history_' + name_of_stock + '_StockVar' + str(stock_variable_index)  + '_Window' + str(window_size_of_observation_for_prediction)
                history = load_data(name_to_load=name_to_save, path=path_save+name_of_stock+'/plots/')
                fig, ax = plt.subplots(nrows=1, ncols=1)  # create figure & 1 axis
                plt.plot(history['loss'])
                plt.ylabel('Loss')
                plt.xlabel('Epoch')
                # plt.show()
                name_of_image = name_to_save
                plt.savefig(path_save + name_of_stock + '/plots/' + name_of_image + '.png')
                # save figure:
                # https://stackoverflow.com/questions/7290370/store-and-reload-matplotlib-pyplot-object?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
                with open(path_save + name_of_stock + '/plots/' + name_of_image + '.pickle', 'wb') as fid:
                    pickle.dump((fig, ax), fid)
                plt.close(fig)  # close the figure

            # ------ time series prediction - testing:
            if predictor_type == 'SVR':
                predicted_variable = SVR_model_list[stock_variable_index].predict(X=X_test)
                predicted_stock = np.hstack([predicted_stock, np.matrix(predicted_variable).T])
            elif predictor_type == 'LSTM':
                X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
                predicted_variable = lstm.predict_point_by_point(LSTM_model_list[stock_variable_index], X_test)
                predicted_stock = np.hstack([predicted_stock, np.matrix(predicted_variable).T])

        # plt.plot(predicted_stock[:, 2], '.', color='k')
        # plt.plot(predicted_stock[:, 3], '.', color='g')
        # plt.plot(predicted_stock[:, 6], '.', color='r')
        # plt.show()
        # input('hi')

        # ------ Denormalization of predicted test samples:
        # https://stackoverflow.com/questions/44552031/sklearnstandardscaler-can-i-inverse-the-standardscaler-for-the-model-output
        #a = np.column_stack((predicted_stock, predicted_stock, predicted_stock, predicted_stock, predicted_stock))
        predicted_testSet_AllStockVariables_unscaled = np.zeros(predicted_stock.shape)
        for test_sample_index in range(Number_of_test_samples):
            predicted_testSet_AllStockVariables_unscaled[test_sample_index, :] = scaler_testing_list[test_sample_index].inverse_transform(X=predicted_stock[test_sample_index, :])

        # plt.plot(predicted_testSet_AllStockVariables_unscaled[:, 2], '.', color='k')
        # plt.plot(predicted_testSet_AllStockVariables_unscaled[:, 3], '.', color='g')
        # # plt.plot(predicted_testSet_AllStockVariables_unscaled[:, 5], '.', color='r')
        # plt.plot(testing_set[:, 6], '.', color='y')
        # plt.show()
        # input('hi')

        actual_testSet_stockVariable_percentage_list = []
        predicted_testSet_stockVariable_percentage_list = []
        actual_AllDataset_stockVariable_percentage_list = []
        actual_AllDataset_stockVariable_percentage_list_averaged = []
        predicted_time_series_list = []  # used for candle plot
        actual_testSet_stockVariable_percentage_not_averaged_list = []
        hit_rate_list = []
        MAE_list = []
        Coefficient_of_Determ_list = []
        RMSE_list = []

        # ------ removing the first row of actual test set and the last row of predicted test set: --> because we are predicting tomorrow
        testing_set = testing_set[1:, :]  # removing first row
        predicted_testSet_AllStockVariables_unscaled = predicted_testSet_AllStockVariables_unscaled[:-1, :]  # removing last row --> today (last day does not have tomorrow and should be removed)

        for stock_variable_index in range(number_of_stock_variables):  # iterating over: Openning, Closing, Low, High, Volume, RSI, ADX, SAR

            # ------ changing to percentage:
            _, _, actual_testSet_stockVariable_percentage = change_timeseries_to_percentage(timeseries=np.array(testing_set[:, stock_variable_index]))
            _, _, predicted_testSet_stockVariable_percentage = change_timeseries_to_percentage(timeseries=np.array(predicted_testSet_AllStockVariables_unscaled[:, stock_variable_index]))
            _, _, actual_AllDataset_stockVariable_percentage = change_timeseries_to_percentage(timeseries=np.array(data_a_stock[:, stock_variable_index]))
            _, _, actual_AllDataset_stockVariable_percentage_averaged = change_timeseries_to_percentage(timeseries=np.array(data_a_stock_averaged[:, stock_variable_index]))
            # ------ saving all features of actual and predictions:
            actual_testSet_stockVariable_percentage_list.append(actual_testSet_stockVariable_percentage)
            predicted_testSet_stockVariable_percentage_list.append(predicted_testSet_stockVariable_percentage)
            actual_AllDataset_stockVariable_percentage_list.append(actual_AllDataset_stockVariable_percentage)
            actual_AllDataset_stockVariable_percentage_list_averaged.append(actual_AllDataset_stockVariable_percentage_averaged)
            actual_AllDatasetset_stockVariable_list.append(data_a_stock[:, stock_variable_index])
            predicted_time_series_list.append(predicted_testSet_AllStockVariables_unscaled[:, stock_variable_index])  # used for candle plot

            # ------ Evaluation metrics for time series prediction:
            # Hit Rate >> Percentage of correct trend prediction:
            hit_rate = 100 * hit_rate_calculation(timeseries_actual_percentage=actual_testSet_stockVariable_percentage,
                                                      timeseries_predicted_percentage=predicted_testSet_stockVariable_percentage)
            hit_rate_list.append(hit_rate)
            name_to_save = 'Hit_rate_' + name_of_stock + '_Window' + str(window_size_of_observation_for_prediction)
            save_data(data=hit_rate_list, name_to_save=name_to_save, path=path_save+name_of_stock+'/')
            save_np_array_to_txt(variable=np.array(hit_rate_list), name_of_variable=name_to_save, path_to_save=path_save+name_of_stock+'/')

            # Mean Absolute Error calculation:
            MAE = sum(abs(actual_testSet_stockVariable_percentage - predicted_testSet_stockVariable_percentage)) / len(predicted_testSet_stockVariable_percentage)
            MAE_list.append(MAE)
            name_to_save = 'Mean_Abs_Error_' + name_of_stock + '_Window' + str(window_size_of_observation_for_prediction)
            save_data(data=MAE_list, name_to_save=name_to_save, path=path_save+name_of_stock+'/')
            save_np_array_to_txt(variable=np.array(MAE_list), name_of_variable=name_to_save, path_to_save=path_save+name_of_stock+'/')

            # Coefficient of Determination:
            r2 = 1 - (sum(np.power(actual_testSet_stockVariable_percentage - predicted_testSet_stockVariable_percentage, 2))) / (
                sum(np.power(actual_testSet_stockVariable_percentage - np.mean(actual_testSet_stockVariable_percentage), 2)))
            Coefficient_of_Determ_list.append(r2)
            name_to_save = 'Coefficieent_of_Determinent_' + name_of_stock + '_Window' + str(window_size_of_observation_for_prediction)
            save_data(data=Coefficient_of_Determ_list, name_to_save=name_to_save, path=path_save+name_of_stock+'/')
            save_np_array_to_txt(variable=np.array(Coefficient_of_Determ_list), name_of_variable=name_to_save, path_to_save=path_save+name_of_stock+'/')

            # Root Mean Square Error calculation:
            RMSE = np.sqrt(sum(np.power((actual_testSet_stockVariable_percentage - predicted_testSet_stockVariable_percentage), 2) / len(predicted_testSet_stockVariable_percentage)))
            RMSE_list.append(RMSE)
            name_to_save = 'Root_Mean_Square_Error_' + name_of_stock + '_Window' + str(window_size_of_observation_for_prediction)
            save_data(data=RMSE_list, name_to_save=name_to_save, path=path_save+name_of_stock+'/')
            save_np_array_to_txt(variable=np.array(RMSE_list), name_of_variable=name_to_save, path_to_save=path_save+name_of_stock+'/')

            # ------ plot the predicted prices:
            if do_plot and stock_variable_index == predict_which_stock_variable:
                name_of_image = 'predicted_dollor_' + name_of_stock + '_Window' + str(window_size_of_observation_for_prediction)
                plot_time_series(time_series_list=[testing_set_not_averaged[:, stock_variable_index], testing_set[:, stock_variable_index],  predicted_testSet_AllStockVariables_unscaled[:, stock_variable_index]], legends=['Actual price', '50-day Moving Average', '50-day Predicted'], path_save=path_save+name_of_stock+'/plots/', name_of_image=name_of_image, format_of_image='png')
                name_of_image = 'predicted_percent_' + name_of_stock + '_Window' + str(window_size_of_observation_for_prediction)
                plot_time_series(time_series_list=[actual_testSet_stockVariable_percentage, predicted_testSet_stockVariable_percentage], legends=['Actual', 'Predicted'], path_save=path_save+name_of_stock+'/plots/', name_of_image=name_of_image, format_of_image='png')

            print('hit_rate of ' + str(stock_variable_index) + ' price of stock ' + str(name_of_stock) + ': ' + str(hit_rate_list[-1]) + '%')
            print('MAE of ' +str(stock_variable_index) + ' price of stock ' + str(name_of_stock) + ': ' + str(MAE_list[-1]))
            print('RMSE_list of ' + str(stock_variable_index) + ' price of stock ' +str(name_of_stock) + ': ' + str(RMSE_list[-1]))
            print('Coefficient_of_Determ_list of ' + str(stock_variable_index) + ' price of stock ' +str(name_of_stock) + ': ' + str(Coefficient_of_Determ_list[-1]))
            print('  ')

            # ------ plot the time series prediction:
            save_data(data=predicted_testSet_AllStockVariables_unscaled[:, stock_variable_index], name_to_save='predicted_testSet_StockVariable' + str(stock_variable_index) + '_unscaled' + '_Window' + str(window_size_of_observation_for_prediction),
                      path=path_save + 'time_series_prediction/' + str(name_of_stock) + '/')
            save_np_array_to_txt(variable=np.array(predicted_testSet_AllStockVariables_unscaled[:, stock_variable_index]),
                                 name_of_variable='predicted_testSet_StockVariable' + str(stock_variable_index) + '_unscaled' + '_Window' + str(window_size_of_observation_for_prediction),
                                 path_to_save=path_save + 'time_series_prediction/' + str(name_of_stock) + '/')
            save_data(data=testing_set[:, stock_variable_index],
                      name_to_save='testing_set_StockVariable' + str(stock_variable_index) + '_Window' + str(window_size_of_observation_for_prediction),
                      path=path_save + 'time_series_prediction/' + str(name_of_stock) + '/')
            save_np_array_to_txt(variable=np.array(testing_set[:, stock_variable_index]),
                                 name_of_variable='testing_set_StockVariable' + str(stock_variable_index) + '_Window' + str(window_size_of_observation_for_prediction),
                                 path_to_save=path_save + 'time_series_prediction/'  + str(name_of_stock) + '/')
            save_data(data=actual_testSet_stockVariable_percentage,
                      name_to_save='actual_testSet_stockVariable' + str(stock_variable_index) + '_percentage' + '_Window' + str(window_size_of_observation_for_prediction),
                      path=path_save + 'time_series_prediction/' + str(name_of_stock) + '/')
            save_np_array_to_txt(variable=np.array(actual_testSet_stockVariable_percentage),
                                 name_of_variable='actual_testSet_stockVariable' + str(stock_variable_index) + '_percentage' + '_Window' + str(window_size_of_observation_for_prediction),
                                 path_to_save=path_save + 'time_series_prediction/' + str(name_of_stock) + '/')
            save_data(data=predicted_testSet_stockVariable_percentage,
                      name_to_save='predicted_testSet_stockVariable' + str(stock_variable_index) + '_percentage' + '_Window' + str(window_size_of_observation_for_prediction),
                      path=path_save + 'time_series_prediction/' + str(name_of_stock) + '/')
            save_np_array_to_txt(variable=np.array(predicted_testSet_stockVariable_percentage),
                                 name_of_variable='predicted_testSet_stockVariable' + str(stock_variable_index) + '_percentage' + '_Window' + str(window_size_of_observation_for_prediction),
                                 path_to_save=path_save + 'time_series_prediction/' + str(name_of_stock) + '/')

        hit_rate_all_stocks = np.vstack([hit_rate_all_stocks, hit_rate_list])
        MAE_all_stocks = np.vstack([MAE_all_stocks, MAE_list])
        RMSE_all_stocks = np.vstack([RMSE_all_stocks, RMSE_list])


        #------ plot candle plot:
        if do_plot:
            name_of_image = 'predicted_dollor_CANDLE_' + name_of_stock
            plot_candle(time_series_list=actual_AllDatasetset_stockVariable_list, path_save=path_save+name_of_stock+'/plots/', name_of_image=name_of_image, format_of_image='png')

        # ------ take only one of the predicted variables:
        actual_testSet_targetStockVariable_percentage = actual_testSet_stockVariable_percentage_list[predict_which_stock_variable]
        predicted_testSet_targetStockVariable_percentage = predicted_testSet_stockVariable_percentage_list[predict_which_stock_variable]
        actual_AllDataset_targetStockVariable_percentage = actual_AllDataset_stockVariable_percentage_list[predict_which_stock_variable]
        actual_AllDataset_targetStockVariable_percentage_averaged = actual_AllDataset_stockVariable_percentage_list_averaged[predict_which_stock_variable]

        # ------ save the desired predicted variable of every stock:
        actual_testSet_AllStocks_targetStockVariable_percentage.append(actual_testSet_targetStockVariable_percentage)
        predicted_testSet_AllStocks_targetStockVariable_percentage.append(predicted_testSet_targetStockVariable_percentage)
        actual_AllDataset_AllStocks_targetStockVariable_percentage.append(actual_AllDataset_targetStockVariable_percentage)
        actual_AllDataset_AllStocks_targetStockVariable_percentage_averaged.append(actual_AllDataset_targetStockVariable_percentage_averaged)

        # ----- saving actual value of tomorrow
        actual_testingSet_AllStocks_targetStockVariable.append(testing_set_not_averaged[:, predict_which_stock_variable])


        # ADX_plot, = plt.plot(ADX, '-', color='k', label='ADX', linewidth=2.0)
        # SPDI_plot, = plt.plot(SPDI, '-', color='g', label='SPDI', linewidth=1.0)
        # NPDI_plot, = plt.plot(SNDI, '-', color='r', label='SMDI', linewidth=1.0)
        # plt.xlabel('Day')
        # plt.ylabel('Price')
        # plt.legend(handles=[ADX_plot, SPDI_plot, NPDI_plot])
        # plt.show()
        # input('hi')
    hit_rate_average_over_stocks = hit_rate_all_stocks.mean(axis=0)
    hit_rate_std_over_stocks = hit_rate_all_stocks.std(axis=0)
    save_np_array_to_txt(variable=hit_rate_average_over_stocks,
                         name_of_variable='hit_rate_average_over_stocks',
                         path_to_save=path_save + 'time_series_prediction/')
    save_np_array_to_txt(variable=hit_rate_all_stocks,
                         name_of_variable='hit_rate_all_stocks',
                         path_to_save=path_save + 'time_series_prediction/')
    save_np_array_to_txt(variable=MAE_all_stocks,
                         name_of_variable='MAE_all_stocks',
                         path_to_save=path_save + 'time_series_prediction/')
    save_np_array_to_txt(variable=RMSE_all_stocks,
                         name_of_variable='RMSE_all_stocks',
                         path_to_save=path_save + 'time_series_prediction/')
    save_np_array_to_txt(variable=hit_rate_std_over_stocks,
                         name_of_variable='hit_rate_std_over_stocks',
                         path_to_save=path_save + 'time_series_prediction/')

    if do_suggest_weights:

        weights_obtained_from_portfolio_list = []
        weights_obtained_from_fuzzy_total_list = []
        weights_obtained_from_fuzzy_fundamental_list = []
        weights_obtained_from_fuzzy_technical_list = []
        weights_obtained_from_random_list = []

        actual_profit_tomorrow_portfolio_list = []
        actual_Budget_portfolio_list = [Initial_Budget]
        predicted_Budget_portfolio_list = [Initial_Budget]
        predicted_upper_bound_Budget_portfolio_list = [Initial_Budget]
        predicted_lower_bound_Budget_portfolio_list = [Initial_Budget]
        actual_Budget_portfolio = Initial_Budget
        predicted_Budget_portfolio = Initial_Budget
        predicted_upper_bound_Budget_portfolio = Initial_Budget
        predicted_lower_bound_Budget_portfolio = Initial_Budget
        portfolio_optimum_expected_return_list = []
        portfolio_optimum_std_list = []

        actual_profit_tomorrow_Fuzzy_list = []
        actual_Budget_Fuzzy_list = [Initial_Budget]
        predicted_Budget_Fuzzy_list = [Initial_Budget]
        predicted_upper_bound_Budget_Fuzzy_list = [Initial_Budget]
        predicted_lower_bound_Budget_Fuzzy_list = [Initial_Budget]
        actual_Budget_Fuzzy = Initial_Budget
        predicted_Budget_Fuzzy = Initial_Budget
        predicted_upper_bound_Budget_Fuzzy = Initial_Budget
        predicted_lower_bound_Budget_Fuzzy = Initial_Budget
        Fuzzy_expected_return_list = []
        Fuzzy_std_list = []

        actual_profit_tomorrow_random_list = []
        actual_Budget_random_list = [Initial_Budget]
        predicted_Budget_random_list = [Initial_Budget]
        predicted_upper_bound_Budget_random_list = [Initial_Budget]
        predicted_lower_bound_Budget_random_list = [Initial_Budget]
        actual_Budget_random = Initial_Budget
        predicted_Budget_random = Initial_Budget
        predicted_upper_bound_Budget_random = Initial_Budget
        predicted_lower_bound_Budget_random = Initial_Budget
        random_expected_return_list = []
        random_std_list = []

        # ------ Iteration on the target day
        target_day_index = 1
        while target_day_index <= target_day_range:
            print('========> Processing day ' + str(target_day_index) + ' out of ' + str(target_day_range) + ' days.....')

            # ------ calculating covariance matrix:
            actual_AllDataset_AllStocks_targetStockVariable_percentage_averaged = np.asarray(actual_AllDataset_AllStocks_targetStockVariable_percentage_averaged)
            target_day_with_offset_of_trainingSet = (target_day_index - 1 + initial_day_in_testSet -1) + training_set.shape[0]
            stocks_information_for_covariance_calculation = actual_AllDataset_AllStocks_targetStockVariable_percentage_averaged[:,target_day_with_offset_of_trainingSet-window_size_covariance_calculation:target_day_with_offset_of_trainingSet]
            covariance_matrix = calculate_covariance_matrix(stocks_information_for_covariance_calculation=stocks_information_for_covariance_calculation)

            # ------ preparing expected returns of target day:
            predicted_tomorrow_percentage = np.empty((0, 1))
            actual_tomorrow_percentage = np.empty((0, 1))
            actual_tomorrow_dollars = np.empty((0, 1))
            for stock_index in range(number_of_stocks):
                predicted_tomorrow_percentage = np.vstack([predicted_tomorrow_percentage, predicted_testSet_AllStocks_targetStockVariable_percentage[stock_index].ravel()[target_day_index - 1 + initial_day_in_testSet - 1]])
                actual_tomorrow_percentage = np.vstack([actual_tomorrow_percentage, actual_AllDataset_AllStocks_targetStockVariable_percentage[stock_index].ravel()[target_day_index - 1 + initial_day_in_testSet - 1]])
                actual_tomorrow_dollars = np.vstack([actual_tomorrow_dollars, actual_testingSet_AllStocks_targetStockVariable[stock_index].ravel()[target_day_index - 1 + initial_day_in_testSet - 1]])


            # ------ Markowitz Portfolio:
            print('Status of code: Portfolio theory.....')
            portfolio = Portfolio(covariance_matrix=covariance_matrix, expected_returns=predicted_tomorrow_percentage, desired_variance=1, risk_tolerance_percentage=risk_tolerance_percentage)
            # ------ creating portfilo frontier curve:
            number_of_portfolio_points = 200000
            expected_return_portfolio_points_list = []
            std_portfolio_points_list = []
            weights_list = []
            for point_index in range(number_of_portfolio_points):
               expected_return_portfilio_points, std_portfolio_points, weights = portfolio.random_portfolio_points()
               expected_return_portfolio_points_list.append(expected_return_portfilio_points)
               std_portfolio_points_list.append(std_portfolio_points)
               weights_list.append(weights)
            # ------ plotting portfilo frontier curve:
            if do_plot:
               fig, ax = plt.subplots(nrows=1, ncols=1)  # create figure & 1 axis
               plt.plot(std_portfolio_points_list, expected_return_portfolio_points_list, '*')
               plt.xlabel('Risk (standard deviation)')
               plt.ylabel('Expected return')
               # plt.show()
               if not os.path.exists(path_save + '/plots/'):
                   os.makedirs(path_save + '/plots/')
               plt.savefig(path_save + '/plots/' + 'portfolio_curve_day' + str(target_day_index) + '.png')
               # save figure:
               # https://stackoverflow.com/questions/7290370/store-and-reload-matplotlib-pyplot-object?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
               with open(path_save + '/plots/' + 'portfolio_curve_day' + str(target_day_index) + '.pickle', 'wb') as fid:
                   pickle.dump((fig, ax), fid)
               plt.close(fig)  # close the figure
           # ------ finding desired point on portfilo frontier curve:
            portfolio_optimum_expected_return, portfolio_optimum_std, portfolio_optimum_weights, risk_tolerance = \
                       portfolio.find_the_desired_optimum_portfolio(expected_return_portfolio_points_list=expected_return_portfolio_points_list,
                                                                    std_portfolio_points_list=std_portfolio_points_list, weights_list=weights_list)

            # ------ plotting desired point on portfilo frontier curve:
            if do_plot:
                fig, ax = plt.subplots(nrows=1, ncols=1)  # create figure & 1 axis
                plt.plot(std_portfolio_points_list, expected_return_portfolio_points_list, '*')
                plt.plot(portfolio_optimum_std, portfolio_optimum_expected_return, 'ro')
                plt.plot([risk_tolerance, risk_tolerance], [min(expected_return_portfolio_points_list), max(expected_return_portfolio_points_list)], 'r--')
                plt.plot([min(std_portfolio_points_list), max(std_portfolio_points_list)], [portfolio_optimum_expected_return, portfolio_optimum_expected_return], 'r--')
                plt.xlabel('Risk (standard deviation)')
                plt.ylabel('Expected return')
                # plt.show()
                plt.savefig(path_save + '/plots/' + 'portfolio_curve_day' + str(target_day_index) + '_desired.png')
                # save figure:
                # https://stackoverflow.com/questions/7290370/store-and-reload-matplotlib-pyplot-object?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
                with open(path_save + '/plots/' + 'portfolio_curve_day' + str(target_day_index) + '_desired' + '.pickle', 'wb') as fid:
                    pickle.dump((fig, ax), fid)
                plt.close(fig)  # close the figure
            # ------ Reporting weights obtained by portfolio theory:
            weights_obtained_from_portfolio = portfolio_optimum_weights

            # ------ Reporting future budgets:
            Budget_portfolio_portions = [actual_Budget_portfolio * x for x in weights_obtained_from_portfolio]
            actual_profit_tomorrow_portfolio = np.dot(list(actual_tomorrow_percentage.ravel()), Budget_portfolio_portions)
            actual_profit_tomorrow_portfolio_list.append(actual_profit_tomorrow_portfolio)
            weights_obtained_from_portfolio_list.append(weights_obtained_from_portfolio)

            portfolio_optimum_expected_return_list.append(portfolio_optimum_expected_return)
            portfolio_optimum_std_list.append(portfolio_optimum_std)

            print('actual tomorrows return:', list(actual_tomorrow_percentage.ravel()))
            print('portfolio weights:', weights_obtained_from_portfolio)
            print('tomorrows profit: ', actual_profit_tomorrow_portfolio)
            print('Budget_portions: ', Budget_portfolio_portions)
            print('Obtained expected return by portfolio theory: ', portfolio_optimum_expected_return)
            print('Obtained standard deviation (risk) by portfolio theory: ', portfolio_optimum_std)

            if portfolio_optimum_expected_return >= 0:  # if the profit is positive
                actual_Budget_portfolio = actual_Budget_portfolio + actual_profit_tomorrow_portfolio   # updating actual budget
                predicted_Budget_portfolio = predicted_Budget_portfolio * (1 + portfolio_optimum_expected_return) # updating predicted budget
                predicted_upper_bound_Budget_portfolio = predicted_upper_bound_Budget_portfolio * (1 + portfolio_optimum_expected_return + portfolio_optimum_std)
                predicted_lower_bound_Budget_portfolio = predicted_lower_bound_Budget_portfolio * (1 + portfolio_optimum_expected_return - portfolio_optimum_std)
            else:
                print('Exiting the stock market for this day....')
                pass   # exit stock market for that day

            actual_Budget_portfolio_list.append(actual_Budget_portfolio)
            predicted_Budget_portfolio_list.append(predicted_Budget_portfolio)
            predicted_upper_bound_Budget_portfolio_list.append(predicted_upper_bound_Budget_portfolio)
            predicted_lower_bound_Budget_portfolio_list.append(predicted_lower_bound_Budget_portfolio)
            print('****************************')

            # ------ Start of FUZZY INVEST COUNSELOR:
            # ------ Reading fundamental dataset:
            data_fundamental_df = pd.read_csv(path_dataset_fundamental, delimiter=',')
            data_fundamental = data_fundamental_df.values  # converting pandas data frame to numpy array
            data_fundamental_of_stocks = []   # a list: whose each element is a list regarding a stock --> every inner list a list regarding the fundamental elements --> every inner inner list is an array regarding fundamental data for 4 years.
                                              # example: data_fundamental_of_stocks[5][3][0] --> stock index: 5, fundamental data index: 3, first (index 0) of that fundamental data
            for name_of_stock in stocks_to_process:
                data_fundamental_a_stock = read_data_of_a_stock(dataset=data_fundamental, name_of_stock=name_of_stock)
                caption_of_features_in_dataset = list(data_fundamental_df.columns)
                data_fundamental_a_stock = extract_fundamental_features_from_dataset(dataset=data_fundamental_a_stock,
                                                                                     caption_of_features_in_dataset=caption_of_features_in_dataset,
                                                                                     fundamental_features=fundamental_features)
                data_fundamental_of_stocks.append(data_fundamental_a_stock)

            # ------ converting fundamental dataset to rates:
            number_of_fundamental_features = len(fundamental_features)
            fundamental_rate_data_of_stocks = np.zeros((number_of_stocks, number_of_fundamental_features))
            for stock_index in range(number_of_stocks):
                data_fundamental_a_stock = data_fundamental_of_stocks[stock_index]
                for feature_index in range(number_of_fundamental_features):
                    data_of_a_fundamental = data_fundamental_a_stock[feature_index]
                    rate1 = (data_of_a_fundamental[1] - data_of_a_fundamental[0]) / (data_of_a_fundamental[0] + 0.0001)
                    rate2 = (data_of_a_fundamental[2] - data_of_a_fundamental[1]) / (data_of_a_fundamental[1] + 0.0001)
                    rate_mean = np.mean([rate1, rate2])
                    fundamental_rate_data_of_stocks[stock_index, feature_index] = rate_mean

            # ------ Fuzzy invest counselor:
            print('Status of code: Fuzzy invest counselor.....')
            fuzzy_logic = Fuzzy_logic(covariance_matrix=covariance_matrix, expected_returns=predicted_tomorrow_percentage,
                                      risk_tolerance_percentage=risk_tolerance_percentage, fundamental_rate_data_of_stocks=fundamental_rate_data_of_stocks)
            weights_total_fuzzy, weights_technical_fuzzy, weights_technical_matrix_fuzzy, weights_fundamental_fuzzy, expected_return_fuzzy, std_fuzzy = fuzzy_logic.find_optimum_weights_for_investment(include_technical_analysis=include_technical_analysis,
                                                                                                                        include_fundamental_analysis=include_fundamental_analysis)
            # ------ saving Fuzzy results:
            weights_obtained_from_fuzzy_total_list.append(weights_total_fuzzy)
            weights_obtained_from_fuzzy_fundamental_list.append(weights_fundamental_fuzzy)
            weights_obtained_from_fuzzy_technical_list.append(weights_technical_fuzzy)

            Budget_Fuzzy_portions = [actual_Budget_Fuzzy * x for x in weights_total_fuzzy]
            actual_profit_tomorrow_Fuzzy = np.dot(list(actual_tomorrow_percentage.ravel()), Budget_Fuzzy_portions)
            actual_profit_tomorrow_Fuzzy_list.append(actual_profit_tomorrow_Fuzzy)

            Fuzzy_expected_return_list.append(expected_return_fuzzy)
            Fuzzy_std_list.append(std_fuzzy)

            #------ Reporting results by Fuzzy invest counselor:
            print('Total Weights obtained by Fuzzy invest counselor: ', weights_total_fuzzy)
            print('Technical Weights obtained by Fuzzy invest counselor: ', weights_technical_fuzzy)
            print('Fundamental Weights obtained by Fuzzy invest counselor: ', weights_fundamental_fuzzy)
            print('actual tomorrows prices:', list(actual_tomorrow_percentage.ravel()))
            print('tomorrows profit: ', actual_profit_tomorrow_Fuzzy)
            print('Budget_portions: ', Budget_Fuzzy_portions)
            print('Obtained expected return by Fuzzy invest counselor: ', expected_return_fuzzy)
            print('Obtained standard deviation (risk) by Fuzzy invest counselor: ', std_fuzzy)

            if expected_return_fuzzy >= 0:  # if the profit is positive
                actual_Budget_Fuzzy = actual_Budget_Fuzzy + actual_profit_tomorrow_Fuzzy  # updating actual budget
                predicted_Budget_Fuzzy = predicted_Budget_Fuzzy * (1 + expected_return_fuzzy)  # updating predicted budget
                predicted_upper_bound_Budget_Fuzzy = predicted_upper_bound_Budget_Fuzzy * (1 + expected_return_fuzzy + std_fuzzy)
                predicted_lower_bound_Budget_Fuzzy = predicted_lower_bound_Budget_Fuzzy * (1 + expected_return_fuzzy - std_fuzzy)
            else:
                print('Exiting the stock market for this day....')
                pass   # exit stock market for that day

            # ------ saving Fuzzy Budgets:
            actual_Budget_Fuzzy_list.append(actual_Budget_Fuzzy)
            predicted_Budget_Fuzzy_list.append(predicted_Budget_Fuzzy)
            predicted_upper_bound_Budget_Fuzzy_list.append(predicted_upper_bound_Budget_Fuzzy)
            predicted_lower_bound_Budget_Fuzzy_list.append(predicted_lower_bound_Budget_Fuzzy)
            print('****************************')

            #------ Random weights:
            random_weights = np.random.rand(number_of_stocks)
            random_weights = random_weights / sum(random_weights)
            random_weights = np.matrix(random_weights).T
            expected_return_random = np.matrix((predicted_tomorrow_percentage)).T * random_weights
            std_random = np.sqrt(random_weights.T * covariance_matrix * random_weights)
            random_weights = np.array(random_weights).ravel().tolist()  # changing matrix(vector) w to a list

            # ------ Reporting results obtained by random weights investment:
            Budget_random_portions = [actual_Budget_random * x for x in random_weights]
            actual_profit_tomorrow_random = np.dot(list(actual_tomorrow_percentage.ravel()), Budget_random_portions)
            actual_profit_tomorrow_random_list.append(actual_profit_tomorrow_random)
            weights_obtained_from_random_list.append(random_weights)
            random_expected_return_list.append(expected_return_random)
            random_std_list.append(std_random)

            print('actual tomorrows prices:', list(actual_tomorrow_percentage.ravel()))
            print('random weights:', random_weights)
            print('tomorrows profit: ', actual_profit_tomorrow_random)
            print('Budget_portions: ', Budget_random_portions)
            print('Obtained expected return by random investment: ', expected_return_random)

            if expected_return_random >= 0:  # if the profit is positive
                actual_Budget_random = actual_Budget_random + actual_profit_tomorrow_random  # updating actual budget
                predicted_Budget_random = predicted_Budget_random * (1 + expected_return_random)  # updating predicted budget
                predicted_upper_bound_Budget_random = predicted_upper_bound_Budget_random * (1 + expected_return_random + std_random)
                predicted_lower_bound_Budget_random = predicted_lower_bound_Budget_random * (1 + expected_return_random - std_random)
            else:
                print('Exiting the stock market for this day....')
                pass  # exit stock market for that day

            # ------ saving random Budgets:
            actual_Budget_random_list.append(actual_Budget_random)
            predicted_Budget_random_list.append(predicted_Budget_random)
            predicted_upper_bound_Budget_random_list.append(predicted_upper_bound_Budget_random)
            predicted_lower_bound_Budget_random_list.append(predicted_lower_bound_Budget_random)
            print('****************************')

            # ------ target day increment
            target_day_index = target_day_index + 1


        print('****************************')
        print('actual_Budget_portfolio', actual_Budget_portfolio_list)
        print('predicted_Budget_portfolio', predicted_Budget_portfolio_list)
        print('upper', predicted_upper_bound_Budget_portfolio_list)
        print('lower', predicted_lower_bound_Budget_portfolio_list)
        print('actual_Budget_Fuzzy', actual_Budget_Fuzzy_list)
        print('predicted_Budget_Fuzzy', predicted_Budget_Fuzzy_list)
        print('upper', predicted_upper_bound_Budget_Fuzzy_list)
        print('lower', predicted_lower_bound_Budget_Fuzzy_list)
        print('actual_Budget_random', actual_Budget_random_list)
        print('predicted_Budget_random', predicted_Budget_random_list)
        print('upper', predicted_upper_bound_Budget_random_list)
        print('lower', predicted_lower_bound_Budget_random_list)
        print('tomorrows profit of portfolio:', actual_profit_tomorrow_portfolio_list)
        print('summation of tomorrows profit of portfolio:', sum(actual_profit_tomorrow_portfolio_list))
        print('****************************')

        # ------ save weights:
        save_data(data=weights_obtained_from_portfolio_list, name_to_save='portfolio_weights', path=path_save+'weights/')
        save_np_array_to_txt(variable=np.array(weights_obtained_from_portfolio_list), name_of_variable='portfolio_weights',
                             path_to_save=path_save+'weights/')
        save_data(data=Budget_portfolio_portions, name_to_save='Budget_portfolio_portions', path=path_save+'weights/')
        save_np_array_to_txt(variable=np.array(Budget_portfolio_portions), name_of_variable='Budget_portfolio_portions',
                             path_to_save=path_save+'weights/')
        save_data(data=weights_obtained_from_fuzzy_total_list, name_to_save='fuzzy_total_weights', path=path_save+'weights/')
        save_np_array_to_txt(variable=np.array(weights_obtained_from_fuzzy_total_list), name_of_variable='fuzzy_total_weights',
                             path_to_save=path_save+'weights/')
        save_data(data=Budget_Fuzzy_portions, name_to_save='Budget_Fuzzy_portions', path=path_save+'weights/')
        save_np_array_to_txt(variable=np.array(Budget_Fuzzy_portions), name_of_variable='Budget_Fuzzy_portions',
                             path_to_save=path_save+'weights/')
        save_data(data=weights_obtained_from_fuzzy_technical_list, name_to_save='fuzzy_technical_weights', path=path_save+'weights/')
        save_np_array_to_txt(variable=np.array(weights_obtained_from_fuzzy_technical_list),
                             name_of_variable='fuzzy_technical_weights',
                             path_to_save=path_save+'weights/')
        save_data(data=weights_obtained_from_fuzzy_fundamental_list, name_to_save='fuzzy_fundamental_weights', path=path_save+'weights/')
        save_np_array_to_txt(variable=np.array(weights_obtained_from_fuzzy_fundamental_list),
                             name_of_variable='fuzzy_fundamental_weights',
                             path_to_save=path_save+'weights/')
        save_data(data=weights_obtained_from_random_list, name_to_save='random_weights',
                  path=path_save+'weights/')
        save_np_array_to_txt(variable=np.array(weights_obtained_from_random_list),
                             name_of_variable='random_weights',
                             path_to_save=path_save+'weights/')
        save_data(data=Budget_random_portions, name_to_save='Budget_random_portions', path=path_save+'weights/')
        save_np_array_to_txt(variable=np.array(Budget_random_portions), name_of_variable='Budget_random_portions',
                             path_to_save=path_save+'weights/')

        # ------ save budgets:
        # ------------ save portfolio budgets:
        save_data(data=actual_Budget_portfolio_list, name_to_save='actual_Budget_portfolio_list',
                  path=path_save + 'portfolio/' + 'budgets/')
        save_np_array_to_txt(variable=np.array(actual_Budget_portfolio_list),
                             name_of_variable='actual_Budget_portfolio_list',
                             path_to_save=path_save + 'portfolio/' + 'budgets/')
        save_data(data=predicted_Budget_portfolio_list, name_to_save='predicted_Budget_portfolio_list',
                  path=path_save + 'portfolio/' + 'budgets/')
        save_np_array_to_txt(variable=np.array(predicted_Budget_portfolio_list),
                             name_of_variable='predicted_Budget_portfolio_list',
                             path_to_save=path_save + 'portfolio/' + 'budgets/')
        save_data(data=predicted_lower_bound_Budget_portfolio_list, name_to_save='predicted_lower_bound_Budget_portfolio_list',
                  path=path_save + 'portfolio/' + 'budgets/')
        save_np_array_to_txt(variable=np.array(predicted_lower_bound_Budget_portfolio_list),
                             name_of_variable='predicted_lower_bound_Budget_portfolio_list',
                             path_to_save=path_save + 'portfolio/' + 'budgets/')
        save_data(data=predicted_upper_bound_Budget_portfolio_list, name_to_save='predicted_upper_bound_Budget_portfolio_list',
                  path=path_save + 'portfolio/' + 'budgets/')
        save_np_array_to_txt(variable=np.array(predicted_upper_bound_Budget_portfolio_list),
                             name_of_variable='predicted_upper_bound_Budget_portfolio_list',
                             path_to_save=path_save + 'portfolio/' + 'budgets/')
        # ------------ save Fuzzy budgets:
        save_data(data=actual_Budget_Fuzzy_list, name_to_save='actual_Budget_Fuzzy_list',
                  path=path_save + 'Fuzzy/' + 'budgets/')
        save_np_array_to_txt(variable=np.array(actual_Budget_Fuzzy_list),
                             name_of_variable='actual_Budget_Fuzzy_list',
                             path_to_save=path_save + 'Fuzzy/' + 'budgets/')
        save_data(data=predicted_Budget_Fuzzy_list, name_to_save='predicted_Budget_Fuzzy_list',
                  path=path_save + 'Fuzzy/' + 'budgets/')
        save_np_array_to_txt(variable=np.array(predicted_Budget_Fuzzy_list),
                             name_of_variable='predicted_Budget_Fuzzy_list',
                             path_to_save=path_save + 'Fuzzy/' + 'budgets/')
        save_data(data=predicted_lower_bound_Budget_Fuzzy_list,
                  name_to_save='predicted_lower_bound_Budget_Fuzzy_list',
                  path=path_save + 'Fuzzy/' + 'budgets/')
        save_np_array_to_txt(variable=np.array(predicted_lower_bound_Budget_Fuzzy_list),
                             name_of_variable='predicted_lower_bound_Budget_Fuzzy_list',
                             path_to_save=path_save + 'Fuzzy/' + 'budgets/')
        save_data(data=predicted_upper_bound_Budget_Fuzzy_list,
                  name_to_save='predicted_upper_bound_Budget_Fuzzy_list',
                  path=path_save + 'Fuzzy/' + 'budgets/')
        save_np_array_to_txt(variable=np.array(predicted_upper_bound_Budget_Fuzzy_list),
                             name_of_variable='predicted_upper_bound_Budget_Fuzzy_list',
                             path_to_save=path_save + 'Fuzzy/' + 'budgets/')
        # ------------ save random budgets:
        save_data(data=actual_Budget_random_list, name_to_save='actual_Budget_random_list',
                  path=path_save + 'random/' + 'budgets/')
        save_np_array_to_txt(variable=np.array(actual_Budget_random_list),
                             name_of_variable='actual_Budget_random_list',
                             path_to_save=path_save + 'random/' + 'budgets/')
        save_data(data=predicted_Budget_random_list, name_to_save='predicted_Budget_random_list',
                  path=path_save + 'random/' + 'budgets/')
        save_np_array_to_txt(variable=np.array(predicted_Budget_random_list),
                             name_of_variable='predicted_Budget_random_list',
                             path_to_save=path_save + 'random/' + 'budgets/')
        save_data(data=predicted_lower_bound_Budget_random_list,
                  name_to_save='predicted_lower_bound_Budget_random_list',
                  path=path_save + 'random/' + 'budgets/')
        save_np_array_to_txt(variable=np.array(predicted_lower_bound_Budget_random_list),
                             name_of_variable='predicted_lower_bound_Budget_random_list',
                             path_to_save=path_save + 'random/' + 'budgets/')
        save_data(data=predicted_upper_bound_Budget_random_list,
                  name_to_save='predicted_upper_bound_Budget_random_list',
                  path=path_save + 'random/' + 'budgets/')
        save_np_array_to_txt(variable=np.array(predicted_upper_bound_Budget_random_list),
                             name_of_variable='predicted_upper_bound_Budget_random_list',
                             path_to_save=path_save + 'random/' + 'budgets/')

        # ------ save expected returns and std:
        # ------------ save portfolio expected returns and std:
        save_data(data=portfolio_optimum_expected_return_list, name_to_save='portfolio_optimum_expected_return_list',
                  path=path_save + 'portfolio/' + 'expected_and_std/')
        save_np_array_to_txt(variable=np.array(portfolio_optimum_expected_return_list),
                             name_of_variable='portfolio_optimum_expected_return_list',
                             path_to_save=path_save + 'portfolio/' + 'expected_and_std/')
        save_data(data=portfolio_optimum_std_list, name_to_save='portfolio_optimum_std_list',
                  path=path_save + 'portfolio/' + 'expected_and_std/')
        save_np_array_to_txt(variable=np.array(portfolio_optimum_std_list),
                             name_of_variable='portfolio_optimum_std_list',
                             path_to_save=path_save + 'portfolio/' + 'expected_and_std/')
        # ------------ save Fuzzy expected returns and std:
        save_data(data=Fuzzy_expected_return_list, name_to_save='Fuzzy_expected_return_list',
                  path=path_save + 'Fuzzy/' + 'expected_and_std/')
        save_np_array_to_txt(variable=np.array(Fuzzy_expected_return_list),
                             name_of_variable='Fuzzy_expected_return_list',
                             path_to_save=path_save + 'Fuzzy/' + 'expected_and_std/')
        save_data(data=Fuzzy_std_list, name_to_save='Fuzzy_std_list',
                  path=path_save + 'Fuzzy/' + 'expected_and_std/')
        save_np_array_to_txt(variable=np.array(Fuzzy_std_list),
                             name_of_variable='Fuzzy_std_list',
                             path_to_save=path_save + 'Fuzzy/' + 'expected_and_std/')
        # ------------ save random expected returns and std:
        save_data(data=random_expected_return_list, name_to_save='random_expected_return_list',
                  path=path_save + 'random/' + 'expected_and_std/')
        save_np_array_to_txt(variable=np.array(random_expected_return_list),
                             name_of_variable='random_expected_return_list',
                             path_to_save=path_save + 'random/' + 'expected_and_std/')
        save_data(data=random_std_list, name_to_save='random_std_list',
                  path=path_save + 'random/' + 'expected_and_std/')
        save_np_array_to_txt(variable=np.array(random_std_list),
                             name_of_variable='random_std_list',
                             path_to_save=path_save + 'random/' + 'expected_and_std/')

        # ------ plotting actual budget and predicted budget
        actual_Budget_portfolio_list = load_data(name_to_load='actual_Budget_portfolio_list',
                                                 path=path_save + 'portfolio/' + 'budgets/')
        predicted_Budget_portfolio_list = load_data(name_to_load='predicted_Budget_portfolio_list',
                                                    path=path_save + 'portfolio/' + 'budgets/')
        predicted_lower_bound_Budget_portfolio_list = load_data(
            name_to_load='predicted_lower_bound_Budget_portfolio_list',
            path=path_save + 'portfolio/' + 'budgets/')
        predicted_upper_bound_Budget_portfolio_list = load_data(
            name_to_load='predicted_upper_bound_Budget_portfolio_list',
            path=path_save + 'portfolio/' + 'budgets/')
        actual_Budget_Fuzzy_list = load_data(name_to_load='actual_Budget_Fuzzy_list',
                                             path=path_save + 'Fuzzy/' + 'budgets/')
        predicted_Budget_Fuzzy_list = load_data(name_to_load='predicted_Budget_Fuzzy_list',
                                                path=path_save + 'Fuzzy/' + 'budgets/')
        predicted_upper_bound_Budget_Fuzzy_list = load_data(name_to_load='predicted_upper_bound_Budget_Fuzzy_list',
                                                            path=path_save + 'Fuzzy/' + 'budgets/')
        predicted_lower_bound_Budget_Fuzzy_list = load_data(name_to_load='predicted_lower_bound_Budget_Fuzzy_list',
                                                            path=path_save + 'Fuzzy/' + 'budgets/')

        name_of_image = 'Budgets_1'
        plot_color_between_lines(time_series_list=[actual_Budget_portfolio_list, predicted_Budget_portfolio_list,
                                                   predicted_lower_bound_Budget_portfolio_list,
                                                   predicted_upper_bound_Budget_portfolio_list,
                                                   actual_Budget_Fuzzy_list, predicted_Budget_Fuzzy_list,
                                                   predicted_upper_bound_Budget_Fuzzy_list,
                                                   predicted_lower_bound_Budget_Fuzzy_list, actual_Budget_random_list,
                                                   predicted_Budget_random_list,
                                                   predicted_lower_bound_Budget_random_list,
                                                   predicted_upper_bound_Budget_random_list],
                                 path_save=path_save + '/plots/', name_of_image=name_of_image, format_of_image='png')
        name_of_image = 'Budgets_2'
        plot_color_between_lines_2(time_series_list=[actual_Budget_portfolio_list, predicted_Budget_portfolio_list,
                                                     predicted_lower_bound_Budget_portfolio_list,
                                                     predicted_upper_bound_Budget_portfolio_list,
                                                     actual_Budget_Fuzzy_list, predicted_Budget_Fuzzy_list,
                                                     predicted_upper_bound_Budget_Fuzzy_list,
                                                     predicted_lower_bound_Budget_Fuzzy_list,
                                                     actual_Budget_random_list],
                                   path_save=path_save + '/plots/', name_of_image=name_of_image, format_of_image='png')


# ------ Functions:
def read_data_of_a_stock(dataset, name_of_stock):
    # taking the column of dataset having the name of stocks:
    names_of_stocks = dataset[:, 1]
    # find the indices of a specific stock:
    stock_index = np.where(names_of_stocks == name_of_stock)
    dataset_selected_stock = dataset[stock_index[0], :]
    return dataset_selected_stock

def extract_features_from_dataset(dataset):
    dataset_only_features = dataset[:, 2:]
    return dataset_only_features

def extract_fundamental_features_from_dataset(dataset, caption_of_features_in_dataset, fundamental_features):
    data_fundamental_a_stock = []
    for fundamental_feature in fundamental_features:
        index_of_fundamental_feature = np.where(np.array(caption_of_features_in_dataset) == fundamental_feature)
        index_of_fundamental_feature = index_of_fundamental_feature[0][0]
        data_fundamental_a_stock.append(dataset[:, index_of_fundamental_feature])
    return data_fundamental_a_stock

def divide_dataset_to_training_and_testing_sets(dataset, fraction_of_training_set):
    number_of_samples = dataset.shape[0]
    number_of_training_samples = int(fraction_of_training_set * number_of_samples)
    training_set = dataset[:number_of_training_samples, :]
    testing_set = dataset[number_of_training_samples:, :]
    return training_set, testing_set

def save_data(data, name_to_save, path):
    if not os.path.exists(path):
        os.makedirs(path)
    with open(path+name_to_save+'.pickle', 'wb') as handle:
        pickle.dump(data, handle)

def load_data(name_to_load, path):
    with open(path+name_to_load+'.pickle', 'rb') as handle:
        data = pickle.load(handle)
    return data

def change_timeseries_to_percentage(timeseries):
    timeseries_today = timeseries[1:]
    timeseries_yesterday = timeseries[:-1]
    timeseries_percentage = (timeseries_today - timeseries_yesterday) / timeseries_yesterday
    return timeseries_today, timeseries_yesterday, timeseries_percentage

def hit_rate_calculation(timeseries_actual_percentage, timeseries_predicted_percentage):

    hit_rate = sum(np.sign(timeseries_actual_percentage) == np.sign(timeseries_predicted_percentage)) / \
               len(timeseries_predicted_percentage)
    return hit_rate

def save_np_array_to_txt(variable, name_of_variable, path_to_save='./'):
    # https://stackoverflow.com/questions/22821460/numpy-save-2d-array-to-text-file/22822701
    if not os.path.exists(path_to_save):  # https://stackoverflow.com/questions/273192/how-can-i-create-a-directory-if-it-does-not-exist
        os.makedirs(path_to_save)
    file_address = path_to_save + name_of_variable + '.txt'
    np.set_printoptions(threshold=np.inf, linewidth=np.inf)  # turn off summarization, line-wrapping
    with open(file_address, 'w') as f:
        f.write(np.array2string(variable, separator=', '))

def calculate_covariance_matrix(stocks_information_for_covariance_calculation):
    number_of_days = stocks_information_for_covariance_calculation.shape[1]
    mean = stocks_information_for_covariance_calculation.mean(axis=1)
    C = 0
    for day_index in range(number_of_days):
        x = np.matrix(stocks_information_for_covariance_calculation[:, day_index] - mean)
        C += np.multiply(x, x.T)
    C = C / number_of_days
    C = np.matrix(C, dtype='float')
    return C

def plot_time_series(time_series_list, legends, path_save='./', name_of_image='img', format_of_image='png'):
    if not os.path.exists(path_save):
        os.makedirs(path_save)
    fig, ax = plt.subplots(nrows=1, ncols=1)  # create figure & 1 axis
    number_of_time_series = len(time_series_list)
    plt_fig = [None] * number_of_time_series
    for index in range(number_of_time_series):
        time_series = time_series_list[index]
        plt_fig[index], = plt.plot(range(len(time_series)), time_series, '-', label=legends[index])
    plt.xlabel('Day')
    plt.ylabel('Price')
    plt.legend(handles=plt_fig)
    # plt.show()
    plt.savefig(path_save+name_of_image+'.'+format_of_image) #, dpi=1000
    # save figure:
    # https://stackoverflow.com/questions/7290370/store-and-reload-matplotlib-pyplot-object?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
    with open(path_save + name_of_image + '.pickle', 'wb') as fid:
        pickle.dump((fig, ax), fid)

    plt.close(fig)  # close the figure

def plot_candle(time_series_list, path_save='./', name_of_image='img', format_of_image='png'):
    # source of code: https://stackoverflow.com/questions/36334665/how-to-plot-ohlc-candlestick-with-datetime-in-matplotlib
    # source of code: https://stackoverflow.com/questions/32292221/matplotlib-finance-candlestick2-ohlc-vertical-line-color-and-width
    # source of code: https://matplotlib.org/examples/pylab_examples/finance_demo.html
    # source of code: https://plot.ly/python/candlestick-charts/
    if not os.path.exists(path_save):
        os.makedirs(path_save)
    fig, ax = plt.subplots()  # create figure & 1 axis
    open_data = time_series_list[0]
    close_data = time_series_list[1]
    low_data = time_series_list[2]
    high_data = time_series_list[3]
    trace = candlestick2_ohlc(ax,
                           open_data,
                           high_data,
                           low_data,
                           close_data,
                           width=0.6,
                           colorup='g', colordown='r')
    SAR_data = time_series_list[-1]
    ADX_data = time_series_list[-2]
    SAR_data_plot, = plt.plot(SAR_data, 'o', label = 'SAR')
    plt.xlabel('Day')
    plt.ylabel('Price')
    plt.legend(handles=[SAR_data_plot])
    # plt.show()
    plt.savefig(path_save+name_of_image+'.'+format_of_image)

    # save figure:
    # https://stackoverflow.com/questions/7290370/store-and-reload-matplotlib-pyplot-object?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
    with open(path_save + name_of_image + '.pickle', 'wb') as fid:
        pickle.dump((fig, ax), fid)

    plt.close(fig)  # close the figure

def plot_color_between_lines(time_series_list, path_save='./', name_of_image='img', format_of_image='png'):
    # https://stackoverflow.com/questions/16417496/matplotlib-fill-between-multiple-lines?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
    #https: // stackoverflow.com / questions / 18386106 / matplotlib - hatched - fill - between - without - edges?utm_medium = organic & utm_source = google_rich_qa & utm_campaign = google_rich_qa
    if not os.path.exists(path_save):
        os.makedirs(path_save)
    fig, ax = plt.subplots(nrows=1, ncols=1)  # create figure & 1 axis
    number_of_time_series = len(time_series_list)
    plt_fig = [None] * number_of_time_series

    # Actual based on portfolio:
    time_series = time_series_list[0]
    plt_fig[0], = plt.plot(range(len(time_series)), time_series,  '-', color='g', label='Actual budget based on portfolio theory')

    # Predicted based on portfolio:
    time_series = time_series_list[1]
    plt_fig[1], = plt.plot(range(len(time_series)), time_series, '--', color='k', label='Predicted budget based on portfolio theory')

    # Lower bound of predicted based on portfolio:
    time_series = time_series_list[2]
    plt_fig[2], = plt.plot(range(len(time_series)), time_series, '-', color='k')

    # Upper bound of predicted based on portfolio:
    time_series = time_series_list[3]
    plt_fig[3], = plt.plot(range(len(time_series)), time_series, '-', color='k')

    # actual Fuzzy:
    time_series = time_series_list[4]
    plt_fig[4], = plt.plot(range(len(time_series)), time_series, '-', color='b', label='Actual budget based on FIC')

    # predicted of Fuzzy:
    time_series = time_series_list[5]
    plt_fig[5], = plt.plot(range(len(time_series)), time_series, '--', color='r', label='Predicted budget based on FIC')

    # Lower bound of predicted of Fuzzy:
    time_series = time_series_list[6]
    plt_fig[6], = plt.plot(range(len(time_series)), time_series, '-', color='r')

    # Upper bound of predicted of Fuzzy:
    time_series = time_series_list[7]
    plt_fig[7], = plt.plot(range(len(time_series)), time_series, '-', color='r')

    # Actual based on random investment:
    time_series = time_series_list[8]
    plt_fig[8], = plt.plot(range(len(time_series)), time_series, '-', color='m', label='Actual budget based on random investment')

    # predicted of random:
    time_series = time_series_list[9]
    plt_fig[9], = plt.plot(range(len(time_series)), time_series, '--', color='c', label='Predicted budget based on random investment')

    # Lower bound of predicted of random:
    time_series = time_series_list[10]
    plt_fig[10], = plt.plot(range(len(time_series)), time_series, '-', color='c')

    # Upper bound of predicted of random:
    time_series = time_series_list[11]
    plt_fig[11], = plt.plot(range(len(time_series)), time_series, '-', color='c')

    # https://matplotlib.org/1.5.3/examples/pylab_examples/hatch_demo.html
    # https://stackoverflow.com/questions/18386106/matplotlib-hatched-fill-between-without-edges?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
    plt.fill_between(range(len(time_series_list[3])), time_series_list[10], time_series_list[11], facecolor='c', alpha=.1, hatch="O", edgecolor="c", linewidth=0.0)
    plt.fill_between(range(len(time_series_list[3])), time_series_list[2],time_series_list[3], color='k', alpha=.1)
    # plt.fill_between(range(len(time_series_list[3])), time_series_list[5], time_series_list[6], facecolor="none",hatch="X",edgecolor="b",linewidth=0.0)
    plt.fill_between(range(len(time_series_list[3])), time_series_list[6], time_series_list[7], facecolor='r', alpha=.1, hatch="X", edgecolor="r", linewidth=0.0)
    plt.xlabel('Day')
    plt.ylabel('Budget (dollars)')
    plt.xticks(range(0, len(time_series), 2))
    plt.legend(handles=plt_fig)
    # plt.show()
    plt.savefig(path_save+name_of_image+'.'+format_of_image)

    # save figure:
    # https://stackoverflow.com/questions/7290370/store-and-reload-matplotlib-pyplot-object?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
    with open(path_save+name_of_image+'.pickle', 'wb') as fid:
        pickle.dump((fig, ax), fid)

    plt.close(fig)  # close the figure

    # this code is for loading the saved figure:
    # with open(path_save+'Budget_.pickle', 'rb') as fid:
    #     fig, ax = pickle.load(fid)
    # plt.show(fig)

def plot_color_between_lines_2(time_series_list, path_save='./', name_of_image='img', format_of_image='png'):
    # https://stackoverflow.com/questions/16417496/matplotlib-fill-between-multiple-lines?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
    #https: // stackoverflow.com / questions / 18386106 / matplotlib - hatched - fill - between - without - edges?utm_medium = organic & utm_source = google_rich_qa & utm_campaign = google_rich_qa
    if not os.path.exists(path_save):
        os.makedirs(path_save)
    fig, ax = plt.subplots(nrows=1, ncols=1)  # create figure & 1 axis
    number_of_time_series = len(time_series_list)
    plt_fig = [None] * number_of_time_series

    # Actual based on portfolio:
    time_series = time_series_list[0]
    plt_fig[0], = plt.plot(range(len(time_series)), time_series,  '-', color='g', label='Actual budget based on portfolio theory')

    # Predicted based on portfolio:
    time_series = time_series_list[1]
    plt_fig[1], = plt.plot(range(len(time_series)), time_series, '--', color='k', label='Predicted budget based on portfolio theory')

    # Lower bound of predicted based on portfolio:
    time_series = time_series_list[2]
    plt_fig[2], = plt.plot(range(len(time_series)), time_series, '-', color='k')

    # Upper bound of predicted based on portfolio:
    time_series = time_series_list[3]
    plt_fig[3], = plt.plot(range(len(time_series)), time_series, '-', color='k')

    # actual Fuzzy:
    time_series = time_series_list[4]
    plt_fig[4], = plt.plot(range(len(time_series)), time_series, '-', color='b', label='Actual budget based on FIC')

    # predicted of Fuzzy:
    time_series = time_series_list[5]
    plt_fig[5], = plt.plot(range(len(time_series)), time_series, '--', color='r', label='Predicted budget based on FIC')

    # Lower bound of predicted of Fuzzy:
    time_series = time_series_list[6]
    plt_fig[6], = plt.plot(range(len(time_series)), time_series, '-', color='r')

    # Upper bound of predicted of Fuzzy:
    time_series = time_series_list[7]
    plt_fig[7], = plt.plot(range(len(time_series)), time_series, '-', color='r')

    # Actual based on random investment:
    time_series = time_series_list[8]
    plt_fig[8], = plt.plot(range(len(time_series)), time_series, '-', color='m', label='Actual budget based on random investment')

    # https://matplotlib.org/1.5.3/examples/pylab_examples/hatch_demo.html
    # https://stackoverflow.com/questions/18386106/matplotlib-hatched-fill-between-without-edges?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
    plt.fill_between(range(len(time_series_list[3])), time_series_list[2],time_series_list[3], color='k', alpha=.1)
    # plt.fill_between(range(len(time_series_list[3])), time_series_list[5], time_series_list[6], facecolor="none",hatch="X",edgecolor="b",linewidth=0.0)
    plt.fill_between(range(len(time_series_list[3])), time_series_list[6], time_series_list[7], facecolor='r', alpha=.1, hatch="X", edgecolor="r", linewidth=0.0)
    plt.xlabel('Day')
    plt.ylabel('Budget (dollars)')
    plt.xticks(range(0, len(time_series), 2))
    plt.legend(handles=plt_fig)
    # plt.show()
    plt.savefig(path_save+name_of_image+'.'+format_of_image)

    # save figure:
    # https://stackoverflow.com/questions/7290370/store-and-reload-matplotlib-pyplot-object?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
    with open(path_save+name_of_image+'.pickle', 'wb') as fid:
        pickle.dump((fig, ax), fid)

    plt.close(fig)  # close the figure

    # this code is for loading the saved figure:
    # with open(path_save+'Budget_.pickle', 'rb') as fid:
    #     fig, ax = pickle.load(fid)
    # plt.show(fig)

def moving_average(data_set, periods):
    # https://gist.github.com/rday/5716218
    weights = np.ones(periods) / periods
    return np.convolve(data_set, weights, mode='valid')

if __name__ == '__main__':
    main()
