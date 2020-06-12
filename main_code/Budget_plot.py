import matplotlib.pyplot as plt
import pickle
import os


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
    # time_series = time_series_list[1]
    # plt_fig[1], = plt.plot(range(len(time_series)), time_series, '--', color='k', label='Predicted budget based on portfolio theory')

    # Lower bound of predicted based on portfolio:
    # time_series = time_series_list[2]
    # plt_fig[2], = plt.plot(range(len(time_series)), time_series, '-', color='k')

    # Upper bound of predicted based on portfolio:
    # time_series = time_series_list[3]
    # plt_fig[3], = plt.plot(range(len(time_series)), time_series, '-', color='k')

    # actual Fuzzy:
    time_series = time_series_list[4]
    plt_fig[4], = plt.plot(range(len(time_series)), time_series, '-', color='b', label='Actual budget based on FIC')

    # predicted of Fuzzy:
    # time_series = time_series_list[5]
    # plt_fig[5], = plt.plot(range(len(time_series)), time_series, '--', color='r', label='Predicted budget based on FIC')

    # Lower bound of predicted of Fuzzy:
    # time_series = time_series_list[6]
    # plt_fig[6], = plt.plot(range(len(time_series)), time_series, '-', color='r')

    # Upper bound of predicted of Fuzzy:
    # time_series = time_series_list[7]
    # plt_fig[7], = plt.plot(range(len(time_series)), time_series, '-', color='r')

    # Actual based on random investment:
    time_series = time_series_list[8]
    plt_fig[8], = plt.plot(range(len(time_series)), time_series, '-', color='m', label='Actual budget based on random investment')

    # predicted of random:
    # time_series = time_series_list[9]
    # plt_fig[9], = plt.plot(range(len(time_series)), time_series, '--', color='c', label='Predicted budget based on random investment')

    # Lower bound of predicted of random:
    # time_series = time_series_list[10]
    # plt_fig[10], = plt.plot(range(len(time_series)), time_series, '-', color='c')

    # Upper bound of predicted of random:
    # time_series = time_series_list[11]
    # plt_fig[11], = plt.plot(range(len(time_series)), time_series, '-', color='c')

    # https://matplotlib.org/1.5.3/examples/pylab_examples/hatch_demo.html
    # https://stackoverflow.com/questions/18386106/matplotlib-hatched-fill-between-without-edges?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
    # plt.fill_between(range(len(time_series_list[3])), time_series_list[10], time_series_list[11], facecolor='c', alpha=.1, hatch="O", edgecolor="c", linewidth=0.0)
    # plt.fill_between(range(len(time_series_list[3])), time_series_list[2],time_series_list[3], color='k', alpha=.1)
    # plt.fill_between(range(len(time_series_list[3])), time_series_list[5], time_series_list[6], facecolor="none",hatch="X",edgecolor="b",linewidth=0.0)
    # plt.fill_between(range(len(time_series_list[3])), time_series_list[6], time_series_list[7], facecolor='r', alpha=.1, hatch="X", edgecolor="r", linewidth=0.0)
    plt.xlabel('Day')
    plt.ylabel('Budget (dollars)')
    plt.xticks(range(0, len(time_series), 2))
    plt.legend(handles=[plt_fig[0], plt_fig[4], plt_fig[8]])
    plt.show()
    # plt.savefig(path_save+name_of_image+'.'+format_of_image)

    # save figure:
    # https://stackoverflow.com/questions/7290370/store-and-reload-matplotlib-pyplot-object?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
    with open(path_save+name_of_image+'.pickle', 'wb') as fid:
        pickle.dump((fig, ax), fid)

    plt.close(fig)  # close the figure

    # this code is for loading the saved figure:
    # with open(path_save+'Budget_1.pickle', 'rb') as fid:
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
    # time_series = time_series_list[1]
    # plt_fig[1], = plt.plot(range(len(time_series)), time_series, '--', color='k', label='Predicted budget based on portfolio theory')

    # Lower bound of predicted based on portfolio:
    # time_series = time_series_list[2]
    # plt_fig[2], = plt.plot(range(len(time_series)), time_series, '-', color='k')

    # Upper bound of predicted based on portfolio:
    # time_series = time_series_list[3]
    # plt_fig[3], = plt.plot(range(len(time_series)), time_series, '-', color='k')

    # actual Fuzzy:
    time_series = time_series_list[4]
    plt_fig[4], = plt.plot(range(len(time_series)), time_series, '-', color='b', label='Actual budget based on FIC')

    # predicted of Fuzzy:
    # time_series = time_series_list[5]
    # plt_fig[5], = plt.plot(range(len(time_series)), time_series, '--', color='r', label='Predicted budget based on FIC')

    # Lower bound of predicted of Fuzzy:
    time_series = time_series_list[6]
    # plt_fig[6], = plt.plot(range(len(time_series)), time_series, '-', color='r')

    # Upper bound of predicted of Fuzzy:
    # time_series = time_series_list[7]
    # plt_fig[7], = plt.plot(range(len(time_series)), time_series, '-', color='r')

    # Actual based on random investment:
    time_series = time_series_list[8]
    plt_fig[8], = plt.plot(range(len(time_series)), time_series, '-', color='m', label='Actual budget based on random investment')

    # https://matplotlib.org/1.5.3/examples/pylab_examples/hatch_demo.html
    # https://stackoverflow.com/questions/18386106/matplotlib-hatched-fill-between-without-edges?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
    # plt.fill_between(range(len(time_series_list[3])), time_series_list[2],time_series_list[3], color='k', alpha=.1)
    # plt.fill_between(range(len(time_series_list[3])), time_series_list[5], time_series_list[6], facecolor="none",hatch="X",edgecolor="b",linewidth=0.0)
    # plt.fill_between(range(len(time_series_list[3])), time_series_list[6], time_series_list[7], facecolor='r', alpha=.1, hatch="X", edgecolor="r", linewidth=0.0)
    plt.xlabel('Day')
    plt.ylabel('Budget (dollars)')
    plt.xticks(range(0, len(time_series), 2))
    plt.legend(handles=[plt_fig[0], plt_fig[4], plt_fig[8]])
    plt.show()
    # plt.savefig(path_save+name_of_image+'.'+format_of_image)

    # save figure:
    # https://stackoverflow.com/questions/7290370/store-and-reload-matplotlib-pyplot-object?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
    # with open(path_save+name_of_image+'.pickle', 'wb') as fid:
    #     pickle.dump((fig, ax), fid)
    #
    # plt.close(fig)  # close the figure

    # this code is for loading the saved figure:
    with open(path_save+'Budget_2.pickle', 'rb') as fid:
         fig, ax = pickle.load(fid)
    plt.show(fig)


def load_data(name_to_load, path):
    with open(path + name_to_load + '.pickle', 'rb') as handle:
        data = pickle.load(handle)
    return data



path_save = './saved_files/'

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
actual_Budget_random_list = load_data(name_to_load='actual_Budget_random_list',
                                     path=path_save + 'random/' + 'budgets/')
predicted_Budget_random_list = load_data(name_to_load='predicted_Budget_random_list',
                                        path=path_save + 'random/' + 'budgets/')
predicted_upper_bound_Budget_random_list = load_data(name_to_load='predicted_upper_bound_Budget_random_list',
                                                    path=path_save + 'random/' + 'budgets/')
predicted_lower_bound_Budget_random_list = load_data(name_to_load='predicted_lower_bound_Budget_random_list',
                                                    path=path_save + 'random/' + 'budgets/')

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
                         path_save=path_save + 'plots/', name_of_image='Budgets_1', format_of_image='png')
name_of_image = 'Budgets_2'
plot_color_between_lines_2(time_series_list=[actual_Budget_portfolio_list, predicted_Budget_portfolio_list,
                                             predicted_lower_bound_Budget_portfolio_list,
                                             predicted_upper_bound_Budget_portfolio_list,
                                             actual_Budget_Fuzzy_list, predicted_Budget_Fuzzy_list,
                                             predicted_upper_bound_Budget_Fuzzy_list,
                                             predicted_lower_bound_Budget_Fuzzy_list,
                                             actual_Budget_random_list],
                           path_save=path_save + 'plots/', name_of_image='Budgets_2', format_of_image='png')

