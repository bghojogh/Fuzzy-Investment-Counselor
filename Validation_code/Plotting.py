import numpy as np
import pandas
import matplotlib.pyplot as plt
import pickle
import os
#from pandas.tools.plotting import parallel_coordinates
from matplotlib import ticker



def main_plotting():
    path_save = './saved_files/'
    stocks_to_process = [ 'GE', 'CAT','GM', 'IBM', 'AAPL']   # 'AAP', 'AAPL', 'ABC', 'ADSK', 'ALXN' # 'AAP', 'AAPL', 'ADSK', 'AIZ', 'ALLE', 'ALXN', 'AMAT', 'AMGN', 'AMZN', 'AN'
    number_of_stocks = len(stocks_to_process)
    window_size_of_observation_for_prediction_list = range(5, 50, 5)
    number_of_stock_variables = 4

    Hit_rate_Allstocks_list_Allvariables = []
    for name_of_stock in stocks_to_process:
        Hit_rate_list_variable = number_of_stock_variables * [None]
        for aa in range(len(Hit_rate_list_variable)):
            Hit_rate_list_variable[aa] = []
        for stock_variable_index in range(number_of_stock_variables):
            print(stock_variable_index)
            for window_size_of_observation_for_prediction in window_size_of_observation_for_prediction_list:
                print('Status of code: Preprocessing & time series forcasting.....')
                name_to_save = 'Hit_rate_' + name_of_stock + '_Window' + str(window_size_of_observation_for_prediction)
                Hit_rate = load_data(name_to_load=name_to_save, path=path_save+name_of_stock+'/')
                Hit_rate_list_variable[stock_variable_index].append(Hit_rate[stock_variable_index])
        Hit_rate_Allstocks_list_Allvariables.append(Hit_rate_list_variable) # A list of HR of different stocks
    Hit_rate_Allstocks_list_Allvariables_average = np.array(Hit_rate_Allstocks_list_Allvariables).mean(axis=1)

    plot_time_series(time_series_list=[Hit_rate_Allstocks_list_Allvariables_average[0], Hit_rate_Allstocks_list_Allvariables_average[1], Hit_rate_Allstocks_list_Allvariables_average[2],
                                       Hit_rate_Allstocks_list_Allvariables_average[3], Hit_rate_Allstocks_list_Allvariables_average[4]],
                                        window_sizes=window_size_of_observation_for_prediction_list,  legends=stocks_to_process, path_save='./saved_files/', name_of_image='Hit_rate', format_of_image='png')


def plot_time_series(time_series_list, legends,  window_sizes, path_save='./', name_of_image='img', format_of_image='png'):
    if not os.path.exists(path_save):
        os.makedirs(path_save)
    fig, ax = plt.subplots(nrows=1, ncols=1)  # create figure & 1 axis
    number_of_time_series = len(time_series_list)
    plt_fig = [None] * number_of_time_series
    for index in range(number_of_time_series):
        if index == 0:
            marker = '-o'
        elif index == 1:
            marker = '-v'
        elif index == 2:
            marker = '-^'
        elif index == 3:
            marker = '-s'
        elif index == 4:
            marker = '-D'
        time_series = time_series_list[index]
        plt_fig[index], = plt.plot(window_sizes, time_series, marker, label=legends[index], linewidth=2.0)
    plt.plot([10, 10], [45, 64], 'r--')
    plt.xlabel(r'Window size ($\Delta_p$)')
    plt.ylabel('Hit rate')
    plt.xticks(window_sizes)
    plt.legend(handles=plt_fig)
    # plt.show()
    plt.savefig(path_save+name_of_image+'.'+format_of_image)

    # save figure:
    # https://stackoverflow.com/questions/7290370/store-and-reload-matplotlib-pyplot-object?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
    with open(path_save + name_of_image + '.pickle', 'wb') as fid:
        pickle.dump((fig, ax), fid)

    plt.close(fig)  # close the figure

def load_data(name_to_load, path):
    with open(path+name_to_load+'.pickle', 'rb') as handle:
        data = pickle.load(handle)
    return data

if __name__ == '__main__':
    main_plotting()