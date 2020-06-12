import numpy as np
import pandas
import matplotlib.pyplot as plt
import pickle
import os
#from pandas.tools.plotting import parallel_coordinates
from matplotlib import ticker
import random
from math import log
from mpl_toolkits import mplot3d
import seaborn as sns



def main_plotting2():
    path_save = './saved_files/'
    stocks_to_process = ['GE', 'CAT','GM', 'IBM', 'AAPL']
    number_of_stocks = len(stocks_to_process)
    window_size_list = range(5, 45, 5)
    c_list = [1000, 100, 10, 0.1] #[0.1, 10, 100, 1000]
    gamma_list = [0.0001, 0.001, 0.01, 0.1] # 0.0001, 0.001, 0.01, 0.1

    optimum_parameters_in_stocks = [[15, 1000, 0.0001],
                                    [15, 1000, 0.0001],
                                    [15, 1000, 0.0001],
                                    [15, 1000, 0.0001],
                                    [15, 1000, 0.0001]]   # order of parameters: window_size, C, gamma

    c_versus_gamma_matrix = np.zeros((len(c_list), len(gamma_list)))
    c_versus_window_matrix = np.zeros((len(c_list), len(window_size_list)))
    gamma_versus_window_matrix = np.zeros((len(gamma_list), len(window_size_list)))
    for stock_index in range(len(stocks_to_process)):
        name_of_stock = stocks_to_process[stock_index]
        for gamma_index in range(len(gamma_list)):
            for c_index in range(len(c_list)):
                window_size = optimum_parameters_in_stocks[stock_index][0]
                mutual_name = '_Window' + str(window_size) + '_c' + str(c_list[c_index]) + '_gamma' + str(gamma_list[gamma_index])
                name_to_save = 'Hit_rate_' + name_of_stock + mutual_name
                Hit_rate = load_data(name_to_load=name_to_save, path=path_save+name_of_stock+'/')
                Hit_rate_average = np.array(Hit_rate).mean()
                c_versus_gamma_matrix[c_index, gamma_index] = Hit_rate_average
        ax = sns.heatmap(c_versus_gamma_matrix, linewidth=0.5, cmap="YlGnBu")
        # plt.xticks(window_size_list)
        labels = [str(x) for x in gamma_list]
        ax.set_xticklabels(labels)
        labels = [str(x) for x in c_list]
        ax.set_yticklabels(labels)
        plt.xlabel(r'$\gamma$')
        plt.ylabel(r'$C$')
        # plt.show()
        name_to_save = 'Hit_rate_c_versus_gamma_matrix' + str(name_of_stock)
        plt.savefig(path_save + name_to_save + '.png')
        # save figure:
        # https://stackoverflow.com/questions/7290370/store-and-reload-matplotlib-pyplot-object?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
        with open(path_save + name_to_save + '.pickle', 'wb') as fid:
            pickle.dump((ax), fid)
        plt.close()

    for stock_index in range(len(stocks_to_process)):
        name_of_stock = stocks_to_process[stock_index]
        for window_size_index in range(len(window_size_list)):
            for c_index in range(len(c_list)):
                gamma = optimum_parameters_in_stocks[stock_index][2]
                mutual_name = '_Window' + str(window_size_list[window_size_index]) + '_c' + str(
                    c_list[c_index]) + '_gamma' + str(gamma)
                name_to_save = 'Hit_rate_' + name_of_stock + mutual_name
                Hit_rate = load_data(name_to_load=name_to_save, path=path_save + name_of_stock + '/')
                Hit_rate_average = np.array(Hit_rate).mean()
                c_versus_window_matrix[c_index, window_size_index] = Hit_rate_average
        ax = sns.heatmap(c_versus_window_matrix, linewidth=0.5, cmap="YlGnBu")
        # plt.xticks(window_size_list)
        labels = [str(x) for x in window_size_list]
        ax.set_xticklabels(labels)
        labels = [str(x) for x in c_list]
        ax.set_yticklabels(labels)
        plt.xlabel(r'Window size ($\Delta_p$)')
        plt.ylabel(r'$C$')
        # plt.show()
        name_to_save = 'Hit_rate_c_versus_window_matrix' + str(name_of_stock)
        plt.savefig(path_save + name_to_save + '.png')
        with open(path_save + name_to_save + '.pickle', 'wb') as fid:
            pickle.dump((ax), fid)
        plt.close()

    for stock_index in range(len(stocks_to_process)):
        name_of_stock = stocks_to_process[stock_index]
        for gamma_index in range(len(gamma_list)):
            for window_size_index in range(len(window_size_list)):
                c = optimum_parameters_in_stocks[stock_index][1]
                mutual_name = '_Window' + str(window_size_list[window_size_index]) + '_c' + str(
                    c) + '_gamma' + str(gamma_list[gamma_index])
                name_to_save = 'Hit_rate_' + name_of_stock + mutual_name
                Hit_rate = load_data(name_to_load=name_to_save, path=path_save + name_of_stock + '/')
                Hit_rate_average = np.array(Hit_rate).mean()
                gamma_versus_window_matrix[gamma_index, window_size_index] = Hit_rate_average
        ax = sns.heatmap(gamma_versus_window_matrix, linewidth=0.5, cmap="YlGnBu")
        # plt.xticks(window_size_list)
        labels = [str(x) for x in window_size_list]
        ax.set_xticklabels(labels)
        labels = [str(x) for x in gamma_list]
        ax.set_yticklabels(labels)
        plt.xlabel(r'Window size ($\Delta_p$)')
        plt.ylabel(r'$\gamma$')
        # plt.show()
        name_to_save = 'Hit_rate_gamma_versus_window_matrix' + str(name_of_stock)
        plt.savefig(path_save + name_to_save + '.png')
        with open(path_save + name_to_save + '.pickle', 'wb') as fid:
            pickle.dump((ax), fid)
        plt.close()

def load_data(name_to_load, path):
    with open(path+name_to_load+'.pickle', 'rb') as handle:
        data = pickle.load(handle)
    return data

def save_data(data, name_to_save, path):
    if not os.path.exists(path):
        os.makedirs(path)
    with open(path+name_to_save+'.pickle', 'wb') as handle:
        pickle.dump(data, handle)

if __name__ == '__main__':
    main_plotting2()