import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt

# inspired by:
# http://pythonhosted.org/scikit-fuzzy/auto_examples/plot_tipping_problem.html#example-plot-tipping-problem-py
# http://pythonhosted.org/scikit-fuzzy/auto_examples/index.html
# https://gist.github.com/mvidalgarcia/5a157afa6b275d058126
# http://nullege.com/codes/search/skfuzzy.trimf


class Fuzzy_logic:


    def __init__(self, covariance_matrix, expected_returns, risk_tolerance_percentage, fundamental_rate_data_of_stocks):
        self.covariance_matrix = covariance_matrix
        self.expected_returns = expected_returns
        self.number_of_stocks = len(expected_returns)
        self.risk_tolerance_percentage = risk_tolerance_percentage
        self.fundamental_rate_data_of_stocks = fundamental_rate_data_of_stocks

    def generate_fuzzy_membership_functions(self, exp_own_range, exp_other_range, std_own_range, std_other_range, corrolation_coefficient_range, risk_tolerance_range, w_range):
        ##### input:
        # membership functions of own expected return:
        exp_own_minus_high_MembershipFunction = fuzz.trimf(exp_own_range, [-1, -1, 0])
        exp_own_low_MembershipFunction = fuzz.trimf(exp_own_range, [0, 0.25, 0.5])
        exp_own_high_MembershipFunction = fuzz.trimf(exp_own_range, [0, 1, 1])
        # membership functions of other expected return:
        exp_other_minus_high_MembershipFunction = fuzz.trimf(exp_other_range, [-1, -1, 0])
        exp_other_low_MembershipFunction = fuzz.trimf(exp_other_range, [0, 0.25, 0.5])
        exp_other_high_MembershipFunction = fuzz.trimf(exp_other_range, [0, 1, 1])
        # membership functions of variances:
        std_own_low_MembershipFunction = fuzz.trimf(std_own_range, [0, 0, 1])
        std_own_medium_MembershipFunction = fuzz.trimf(std_own_range, [0, 0.5, 1])
        std_own_high_MembershipFunction = fuzz.trimf(std_own_range, [0, 1, 1])
        # membership functions of other variances:
        std_other_low_MembershipFunction = fuzz.trimf(std_other_range, [0, 0, 1])
        std_other_medium_MembershipFunction = fuzz.trimf(std_other_range, [0, 0.5, 1])
        std_other_high_MembershipFunction = fuzz.trimf(std_other_range, [0, 1, 1])
        # membership functions of covariances:
        corrolation_coefficient_minus_high_MembershipFunction = fuzz.trimf(corrolation_coefficient_range, [-1, -1, 0])
        corrolation_coefficient_high_MembershipFunction = fuzz.trimf(corrolation_coefficient_range, [0, 1, 1])
        # membership functions of risk tolerance:
        risk_tolerance_low_MembershipFunction = fuzz.trimf(risk_tolerance_range, [0, 0, 0.5])
        risk_tolerance_high_MembershipFunction = fuzz.trimf(risk_tolerance_range, [0.5, 1, 1])
        ##### output:
        # membership functions of output weight:
        w_low_MembershipFunction = fuzz.trimf(w_range, [0, 0, 0.66])
        w_medium_MembershipFunction = fuzz.trimf(w_range, [0.33, 0.5, 0.66])
        w_high_MembershipFunction = fuzz.trimf(w_range, [0.33, 1, 1])
        # self.plot_membership_functions(
        #     membership_functions_list=[w_low_MembershipFunction, w_medium_MembershipFunction,
        #                                w_high_MembershipFunction], range_crisp=w_range, name_of_variable='')
        ##### return of function:
        return exp_own_minus_high_MembershipFunction, exp_own_low_MembershipFunction, exp_own_high_MembershipFunction, \
               exp_other_minus_high_MembershipFunction, exp_other_low_MembershipFunction, exp_other_high_MembershipFunction, std_own_low_MembershipFunction, \
               std_own_medium_MembershipFunction, std_own_high_MembershipFunction, std_other_low_MembershipFunction, \
               std_other_medium_MembershipFunction, std_other_high_MembershipFunction, corrolation_coefficient_minus_high_MembershipFunction, \
               corrolation_coefficient_high_MembershipFunction, risk_tolerance_low_MembershipFunction, \
               risk_tolerance_high_MembershipFunction, w_low_MembershipFunction, w_medium_MembershipFunction, w_high_MembershipFunction

    def plot_membership_functions(self, membership_functions_list, range_crisp, name_of_variable=''):
        for membership_function_index in range(len(membership_functions_list)):
            membership_function = membership_functions_list[membership_function_index]
            plt.plot(range_crisp, membership_function, 'b', linewidth=1.5, label='')
        plt.title(name_of_variable)
        # plt.legend()
        plt.show()

    def fuzzification(self, membership_function, range_crisp_value, crisp_value):
        fuzzified_value = fuzz.interp_membership(range_crisp_value, membership_function, crisp_value)
        return fuzzified_value

    def fuzzy_rules(self, exp_own_minus_high_FUZZIFIED, exp_own_low_FUZZIFIED, exp_own_high_FUZZIFIED,
                    exp_other_minus_high_FUZZIFIED, exp_other_low_FUZZIFIED, exp_other_high_FUZZIFIED,
                    std_own_low_FUZZIFIED, std_own_medium_FUZZIFIED, std_own_high_FUZZIFIED,
                    std_other_low_FUZZIFIED, std_other_medium_FUZZIFIED, std_other_high_FUZZIFIED,
                    corrolation_coefficient_minus_high_FUZZIFIED, corrolation_coefficient_high_FUZZIFIED,
                    risk_tolerance_low_FUZZIFIED, risk_tolerance_high_FUZZIFIED,
                    w_low_MembershipFunction, w_medium_MembershipFunction, w_high_MembershipFunction, Fuzzy_block):
        rules = []
        if Fuzzy_block == 0:  # considering merely own stock
            number_of_rules = 9
            k = 0
            for rule_index in range(number_of_rules):
                if rule_index == k:
                    if_rule = exp_own_minus_high_FUZZIFIED
                    if_rule = np.fmin(if_rule, w_low_MembershipFunction)
                elif rule_index == k+1:
                    if_rule = np.fmin(exp_own_low_FUZZIFIED, std_own_low_FUZZIFIED)
                    if_rule = np.fmin(if_rule, w_medium_MembershipFunction)
                elif rule_index == k+2:
                    if_rule = np.fmax(std_own_medium_FUZZIFIED, std_own_high_FUZZIFIED)
                    if_rule = np.fmin(exp_own_low_FUZZIFIED, if_rule)
                    if_rule = np.fmin(if_rule, risk_tolerance_low_FUZZIFIED)
                    if_rule = np.fmin(if_rule, w_low_MembershipFunction)
                elif rule_index == k+3:
                    if_rule = np.fmax(std_own_medium_FUZZIFIED, std_own_high_FUZZIFIED)
                    if_rule = np.fmin(exp_own_low_FUZZIFIED, if_rule)
                    if_rule = np.fmin(if_rule, risk_tolerance_high_FUZZIFIED)
                    if_rule = np.fmin(if_rule, w_medium_MembershipFunction)
                elif rule_index == k+4:
                    if_rule = np.fmin(exp_own_high_FUZZIFIED, std_own_low_FUZZIFIED)
                    if_rule = np.fmin(if_rule, w_high_MembershipFunction)
                elif rule_index == k+5:
                    if_rule = np.fmin(exp_own_high_FUZZIFIED, std_own_medium_FUZZIFIED)
                    if_rule = np.fmin(if_rule, risk_tolerance_low_FUZZIFIED)
                    if_rule = np.fmin(if_rule, w_medium_MembershipFunction)
                elif rule_index == k+6:
                    if_rule = np.fmin(exp_own_high_FUZZIFIED, std_own_medium_FUZZIFIED)
                    if_rule = np.fmin(if_rule, risk_tolerance_high_FUZZIFIED)
                    if_rule = np.fmin(if_rule, w_high_MembershipFunction)
                elif rule_index == k+7:
                    if_rule = np.fmin(exp_own_high_FUZZIFIED, std_own_high_FUZZIFIED)
                    if_rule = np.fmin(if_rule, risk_tolerance_low_FUZZIFIED)
                    if_rule = np.fmin(if_rule, w_low_MembershipFunction)
                elif rule_index == k+8:
                    if_rule = np.fmin(exp_own_high_FUZZIFIED, std_own_high_FUZZIFIED)
                    if_rule = np.fmin(if_rule, risk_tolerance_high_FUZZIFIED)
                    if_rule = np.fmin(if_rule, w_medium_MembershipFunction)
                rules.append(if_rule)

        elif Fuzzy_block == 1:  # considering merely other stock
            number_of_rules = 16
            k = -9
            for rule_index in range(number_of_rules):
                if rule_index == k+9:
                    if_rule = np.fmin(exp_other_low_FUZZIFIED, corrolation_coefficient_minus_high_FUZZIFIED)
                    if_rule = np.fmin(if_rule, w_low_MembershipFunction)
                elif rule_index == k+10:
                    if_rule = np.fmin(exp_other_low_FUZZIFIED, corrolation_coefficient_high_FUZZIFIED)
                    if_rule = np.fmin(if_rule, std_other_low_FUZZIFIED)
                    if_rule = np.fmin(if_rule, w_medium_MembershipFunction)
                elif rule_index == k+11:
                    if_rule = np.fmin(exp_other_low_FUZZIFIED, corrolation_coefficient_high_FUZZIFIED)
                    if_rule2 = np.fmax(std_own_medium_FUZZIFIED, std_other_high_FUZZIFIED)
                    if_rule = np.fmin(if_rule, if_rule2)
                    if_rule = np.fmin(if_rule, risk_tolerance_low_FUZZIFIED)
                    if_rule = np.fmin(if_rule, w_low_MembershipFunction)
                elif rule_index == k+12:
                    if_rule = np.fmin(exp_other_low_FUZZIFIED, corrolation_coefficient_high_FUZZIFIED)
                    if_rule2 = np.fmax(std_own_medium_FUZZIFIED, std_other_high_FUZZIFIED)
                    if_rule = np.fmin(if_rule, if_rule2)
                    if_rule = np.fmin(if_rule, risk_tolerance_high_FUZZIFIED)
                    if_rule = np.fmin(if_rule, w_medium_MembershipFunction)
                elif rule_index == k+13:
                    if_rule = np.fmin(exp_other_minus_high_FUZZIFIED, corrolation_coefficient_high_FUZZIFIED)
                    if_rule = np.fmin(if_rule, w_low_MembershipFunction)
                elif rule_index == k+14:
                    if_rule = np.fmin(exp_other_high_FUZZIFIED, corrolation_coefficient_minus_high_FUZZIFIED)
                    if_rule = np.fmin(if_rule, w_low_MembershipFunction)
                elif rule_index == k+15:
                    if_rule = np.fmin(exp_other_minus_high_FUZZIFIED, corrolation_coefficient_minus_high_FUZZIFIED)
                    if_rule = np.fmin(if_rule, std_other_low_FUZZIFIED)
                    if_rule = np.fmin(if_rule, w_high_MembershipFunction)
                elif rule_index == k+16:
                    if_rule = np.fmin(exp_other_high_FUZZIFIED, corrolation_coefficient_high_FUZZIFIED)
                    if_rule = np.fmin(if_rule, std_other_low_FUZZIFIED)
                    if_rule = np.fmin(if_rule, w_high_MembershipFunction)
                elif rule_index == k+17:
                    if_rule = np.fmin(exp_other_minus_high_FUZZIFIED, corrolation_coefficient_minus_high_FUZZIFIED)
                    if_rule = np.fmin(if_rule, std_other_medium_FUZZIFIED)
                    if_rule = np.fmin(if_rule, risk_tolerance_low_FUZZIFIED)
                    if_rule = np.fmin(if_rule, w_medium_MembershipFunction)
                elif rule_index == k+18:
                    if_rule = np.fmin(exp_other_minus_high_FUZZIFIED, corrolation_coefficient_minus_high_FUZZIFIED)
                    if_rule = np.fmin(if_rule, std_other_medium_FUZZIFIED)
                    if_rule = np.fmin(if_rule, risk_tolerance_high_FUZZIFIED)
                    if_rule = np.fmin(if_rule, w_high_MembershipFunction)
                elif rule_index == k+19:
                    if_rule = np.fmin(exp_other_high_FUZZIFIED, corrolation_coefficient_high_FUZZIFIED)
                    if_rule = np.fmin(if_rule, std_other_medium_FUZZIFIED)
                    if_rule = np.fmin(if_rule, risk_tolerance_low_FUZZIFIED)
                    if_rule = np.fmin(if_rule, w_medium_MembershipFunction)
                elif rule_index == k+20:
                    if_rule = np.fmin(exp_other_high_FUZZIFIED, corrolation_coefficient_high_FUZZIFIED)
                    if_rule = np.fmin(if_rule, std_other_medium_FUZZIFIED)
                    if_rule = np.fmin(if_rule, risk_tolerance_high_FUZZIFIED)
                    if_rule = np.fmin(if_rule, w_high_MembershipFunction)
                elif rule_index == k+21:
                    if_rule = np.fmin(exp_other_minus_high_FUZZIFIED, corrolation_coefficient_minus_high_FUZZIFIED)
                    if_rule = np.fmin(if_rule, std_other_high_FUZZIFIED)
                    if_rule = np.fmin(if_rule, risk_tolerance_low_FUZZIFIED)
                    if_rule = np.fmin(if_rule, w_low_MembershipFunction)
                elif rule_index == k+22:
                    if_rule = np.fmin(exp_other_minus_high_FUZZIFIED, corrolation_coefficient_minus_high_FUZZIFIED)
                    if_rule = np.fmin(if_rule, std_other_high_FUZZIFIED)
                    if_rule = np.fmin(if_rule, risk_tolerance_high_FUZZIFIED)
                    if_rule = np.fmin(if_rule, w_medium_MembershipFunction)
                elif rule_index == k+23:
                    if_rule = np.fmin(exp_other_high_FUZZIFIED, corrolation_coefficient_high_FUZZIFIED)
                    if_rule = np.fmin(if_rule, std_other_high_FUZZIFIED)
                    if_rule = np.fmin(if_rule, risk_tolerance_low_FUZZIFIED)
                    if_rule = np.fmin(if_rule, w_low_MembershipFunction)
                elif rule_index == k+24:
                    if_rule = np.fmin(exp_other_high_FUZZIFIED, corrolation_coefficient_high_FUZZIFIED)
                    if_rule = np.fmin(if_rule, std_other_high_FUZZIFIED)
                    if_rule = np.fmin(if_rule, risk_tolerance_high_FUZZIFIED)
                    if_rule = np.fmin(if_rule, w_medium_MembershipFunction)
                rules.append(if_rule)

        aggregated_two_rules = rules[0]
        for rule_index in range(1, number_of_rules):
            rule = rules[rule_index]
            aggregated_two_rules = np.fmax(aggregated_two_rules, rule)
        aggregated_rules = aggregated_two_rules
        return aggregated_rules  # note: aggregated_rules is the fuzzified output (which needs to be defuzzified)

    def defuzzification(self, aggregated_rules, range_crisp_value):
        output_defuzzified = fuzz.defuzz(range_crisp_value, aggregated_rules, 'centroid')
        weight = output_defuzzified
        return weight

    def do_Fuzzy_logic(self, exp_own, exp_other, std_own, std_other, corrolation_coefficient, risk_tolerance,
                       exp_own_range, exp_other_range, std_own_range, std_other_range, corrolation_coefficient_range,
                       risk_tolerance_range, w_range, Fuzzy_block):
        ################## generate membership functions:
        exp_own_minus_high_MembershipFunction, exp_own_low_MembershipFunction, exp_own_high_MembershipFunction, \
        exp_other_minus_high_MembershipFunction, exp_other_low_MembershipFunction, exp_other_high_MembershipFunction, std_own_low_MembershipFunction, \
        std_own_medium_MembershipFunction, std_own_high_MembershipFunction, std_other_low_MembershipFunction, \
        std_other_medium_MembershipFunction, std_other_high_MembershipFunction, corrolation_coefficient_minus_high_MembershipFunction, \
        corrolation_coefficient_high_MembershipFunction, risk_tolerance_low_MembershipFunction, \
        risk_tolerance_high_MembershipFunction, w_low_MembershipFunction, w_medium_MembershipFunction, w_high_MembershipFunction = \
            self.generate_fuzzy_membership_functions(exp_own_range, exp_other_range, std_own_range, std_other_range, corrolation_coefficient_range, risk_tolerance_range, w_range)
        ################## Fuzzification:
        #########
        crisp_value = exp_own
        exp_own_minus_high_FUZZIFIED = self.fuzzification(membership_function=exp_own_minus_high_MembershipFunction,
                                             range_crisp_value=exp_own_range, crisp_value=crisp_value)
        #########
        crisp_value = exp_own
        exp_own_low_FUZZIFIED = self.fuzzification(membership_function=exp_own_low_MembershipFunction,
                                                range_crisp_value=exp_own_range, crisp_value=crisp_value)
        #########
        crisp_value = exp_own
        exp_own_high_FUZZIFIED = self.fuzzification(membership_function=exp_own_high_MembershipFunction,
                                                range_crisp_value=exp_own_range, crisp_value=crisp_value)
        #########
        crisp_value = exp_other
        exp_other_minus_high_FUZZIFIED = self.fuzzification(membership_function=exp_other_minus_high_MembershipFunction,
                                                range_crisp_value=exp_other_range, crisp_value=crisp_value)
        #########
        crisp_value = exp_other
        exp_other_low_FUZZIFIED = self.fuzzification(membership_function=exp_other_low_MembershipFunction,
                                                            range_crisp_value=exp_other_range, crisp_value=crisp_value)
        #########
        crisp_value = exp_other
        exp_other_high_FUZZIFIED = self.fuzzification(membership_function=exp_other_high_MembershipFunction,
                                                range_crisp_value=exp_other_range, crisp_value=crisp_value)
        #########
        crisp_value = std_own
        std_own_low_FUZZIFIED = self.fuzzification(membership_function=std_own_low_MembershipFunction,
                                                range_crisp_value=std_own_range, crisp_value=crisp_value)
        #########
        crisp_value = std_own
        std_own_medium_FUZZIFIED = self.fuzzification(membership_function=std_own_medium_MembershipFunction,
                                                    range_crisp_value=std_own_range, crisp_value=crisp_value)
        #########
        crisp_value = std_own
        std_own_high_FUZZIFIED = self.fuzzification(membership_function=std_own_high_MembershipFunction,
                                                range_crisp_value=std_own_range, crisp_value=crisp_value)
        #########
        crisp_value = std_other
        std_other_low_FUZZIFIED = self.fuzzification(membership_function=std_other_low_MembershipFunction,
                                                   range_crisp_value=std_other_range, crisp_value=crisp_value)
        #########
        crisp_value = std_other
        std_other_medium_FUZZIFIED = self.fuzzification(membership_function=std_other_medium_MembershipFunction,
                                                      range_crisp_value=std_other_range, crisp_value=crisp_value)
        #########
        crisp_value = std_other
        std_other_high_FUZZIFIED = self.fuzzification(membership_function=std_other_high_MembershipFunction,
                                                    range_crisp_value=std_other_range, crisp_value=crisp_value)
        #########
        crisp_value = corrolation_coefficient
        corrolation_coefficient_minus_high_FUZZIFIED = self.fuzzification(membership_function=corrolation_coefficient_minus_high_MembershipFunction,
                                                range_crisp_value=corrolation_coefficient_range, crisp_value=crisp_value)
        #########
        crisp_value = corrolation_coefficient
        corrolation_coefficient_high_FUZZIFIED = self.fuzzification(membership_function=corrolation_coefficient_high_MembershipFunction,
                                                range_crisp_value=corrolation_coefficient_range, crisp_value=crisp_value)
        #########
        crisp_value = risk_tolerance
        risk_tolerance_low_FUZZIFIED = self.fuzzification(membership_function=risk_tolerance_low_MembershipFunction,
                                                range_crisp_value=risk_tolerance_range, crisp_value=crisp_value)
        #########
        crisp_value = risk_tolerance
        risk_tolerance_high_FUZZIFIED = self.fuzzification(membership_function=risk_tolerance_high_MembershipFunction,
                                                range_crisp_value=risk_tolerance_range, crisp_value=crisp_value)
        ################## Fuzzy rules:
        aggregated_rules = self.fuzzy_rules(exp_own_minus_high_FUZZIFIED, exp_own_low_FUZZIFIED, exp_own_high_FUZZIFIED,
                         exp_other_minus_high_FUZZIFIED, exp_other_low_FUZZIFIED, exp_other_high_FUZZIFIED,
                         std_own_low_FUZZIFIED, std_own_medium_FUZZIFIED, std_own_high_FUZZIFIED,
                         std_other_low_FUZZIFIED, std_other_medium_FUZZIFIED, std_other_high_FUZZIFIED,
                         corrolation_coefficient_minus_high_FUZZIFIED, corrolation_coefficient_high_FUZZIFIED,
                         risk_tolerance_low_FUZZIFIED, risk_tolerance_high_FUZZIFIED, w_low_MembershipFunction,
                         w_medium_MembershipFunction, w_high_MembershipFunction, Fuzzy_block)
        ################## defuzzification:
        weight = self.defuzzification(aggregated_rules=aggregated_rules, range_crisp_value=w_range)
        return weight

    def find_optimum_weights_for_investment(self, include_technical_analysis=True, include_fundamental_analysis=True):
        number_of_stocks = len(self.expected_returns)
        fundamental_coefficients = np.array([0.3, 0.15, -0.4, -0.5, -0.9])

        if include_technical_analysis:
            print('Technical analysis in Fuzzy logic...')
            # ------ range of variables:
            exp_own_range = np.arange(-1, 1, 0.000001)
            exp_other_range = exp_own_range
            std_own_range = np.arange(0, 1, 0.000001)
            std_other_range = std_own_range
            corrolation_coefficient_range = np.arange(-1, 1, 0.000001)
            risk_tolerance_range = np.arange(0, 1, 0.000001)
            w_range = np.arange(0, 1, 0.000001)
            # ------ Apply Fuzzy logic:
            number_of_Fuzzy_blocks = 2  # 2 because we have one for Exp_own and one for Exp_other
            risk_tolerance = self.risk_tolerance_percentage / 100    # converting from range [0, 100] to [0, 1]
            weights_matrix = [None] * number_of_Fuzzy_blocks
            weights_matrix[0] = np.zeros((number_of_stocks, number_of_stocks))
            weights_matrix[1] = np.zeros((number_of_stocks, number_of_stocks))
            std_own_list = [None]*number_of_stocks
            for stock_index in range(number_of_stocks):
                std_own_list[stock_index] = np.sqrt(self.covariance_matrix[stock_index, stock_index]) / sum(np.sqrt(np.diag(self.covariance_matrix)))
            for Fuzzy_block in range(number_of_Fuzzy_blocks):
                print('Processing Fuzzy block ' + str(Fuzzy_block + 1) + '......')
                corrolation_coefficient_matrix = np.empty((0, number_of_stocks))
                # ------ loop over own stocks (in pair-wise consideration of stocks):
                for stock_index in range(number_of_stocks):
                    print('====> Processing Stock ' + str(stock_index + 1) + '......')
                    exp_own = self.expected_returns[stock_index] / (sum(abs(self.expected_returns))+0.001)
                    # ------ scaling the std_own to range [0,1]:
                    max_std = max(np.array(std_own_list))
                    std_scale = risk_tolerance + max_std * (1 - risk_tolerance)
                    std_own = np.sqrt(self.covariance_matrix[stock_index, stock_index]) / sum(np.sqrt(np.diag(self.covariance_matrix)))
                    std_own = std_own / (std_scale + 0.0001)
                    # ------ creating correlation coefficient out of covariance matrix (excluding self ones):
                    corrolation_coefficient_vector = np.zeros((number_of_stocks))
                    # print('exp_own:', exp_own, 'std_own', std_own)
                    # ------ loop over other stocks (in pair-wise consideration of stocks):
                    for other_stock_index in range(number_of_stocks):
                        if other_stock_index != stock_index:
                            std_other_not_normalized = np.sqrt(self.covariance_matrix[other_stock_index, other_stock_index])
                            std_own_not_normalized = np.sqrt(self.covariance_matrix[stock_index, stock_index])
                            corrolation_coefficient_vector[other_stock_index] = self.covariance_matrix[stock_index, other_stock_index] / (std_own_not_normalized * std_other_not_normalized)
                    corrolation_coefficient_matrix = np.vstack([corrolation_coefficient_matrix, corrolation_coefficient_vector])
                    for other_stock_index in range(number_of_stocks):
                        if other_stock_index != stock_index:
                            exp_other = self.expected_returns[other_stock_index] / (sum(abs(self.expected_returns))+0.001)
                            std_other = np.sqrt(self.covariance_matrix[other_stock_index, other_stock_index]) / sum(np.sqrt(np.diag(self.covariance_matrix)))
                            std_other = std_other / (std_scale + 0.0001)
                            # print('exp_other:', exp_other, 'std_other', std_other)
                            corrolation_coefficient = corrolation_coefficient_vector[other_stock_index]
                            weights_matrix[Fuzzy_block][stock_index, other_stock_index] = self.do_Fuzzy_logic(exp_own=exp_own, exp_other=exp_other, std_own=std_own, std_other=std_other,
                                                                       corrolation_coefficient=corrolation_coefficient, risk_tolerance=risk_tolerance,
                                                                       exp_own_range=exp_own_range, exp_other_range=exp_other_range,
                                                                       std_own_range=std_own_range, std_other_range=std_other_range, corrolation_coefficient_range=corrolation_coefficient_range,
                                                                       risk_tolerance_range=risk_tolerance_range, w_range=w_range, Fuzzy_block=Fuzzy_block)
                            # print(weights_matrix[Fuzzy_block][stock_index, other_stock_index])
                    # print(corrolation_coefficient_matrix)
            # ------ finding the vector of weights of own:
            vector_own_weights = weights_matrix[0].mean(axis=1) * (number_of_stocks / (number_of_stocks - 1))  # because the non-diagonal elements of every row are equal (because we considered only Exp-own for this)
            # ------ extracting the weight matrix considering only 2nd fuzzy block (only Exp_other):
            matrix_other_weights = weights_matrix[1]
            # ------ Multiplyinh matrix of 2nd Fuzzy weights by corrolation_coefficient_matrix element-wise:
            a = np.multiply(matrix_other_weights, abs(corrolation_coefficient_matrix))
            b = a.mean(axis=1) * number_of_stocks
            # ------ Putting vector of own weights on the diag of total weight matrix, and then add the elements in each row:
            weights = risk_tolerance* b + vector_own_weights    # We are trying to emphasize on the importance of risk tolerance
            weights = weights / sum(weights)
            weights_technical = weights
            weights_total = weights_technical
            weights_technical_matrix = weights_matrix

        if include_fundamental_analysis:
            print('Fundamental analysis in Fuzzy logic...')
            fundamental_range = np.arange(-1, 1, 0.000001)
            w_range = np.arange(0, 1, 0.000001)
            fundamental_coefficient_range = np.arange(-1, 1, 0.000001)
            weights_fundamental = []
            for stock_index in range(number_of_stocks):
                fundamental_features = self.fundamental_rate_data_of_stocks[stock_index, :]
                fundamental_features = fundamental_features / (sum(abs(fundamental_features)) + 0.001)
                weights_fundamental.append(self.do_Fuzzy_logic_fundamental(fundamental_range=fundamental_range, w_range=w_range,
                                                  fundamental_coefficient_range=fundamental_coefficient_range,
                                                  fundamental_features=fundamental_features, fundamental_coefficients=fundamental_coefficients))
            weights_fundamental = np.array(weights_fundamental)
            weights_fundamental = weights_fundamental / sum(weights_fundamental)
            weights_total = weights_fundamental

        if (include_technical_analysis is True) and (include_fundamental_analysis is True):
            weights_total = [None] * number_of_stocks
            for stock_index in range(number_of_stocks):
                fundamental_features = self.fundamental_rate_data_of_stocks[stock_index, :]
                number_of_fundamental_features = len(fundamental_features)
                impact_of_fundamental = (np.dot(fundamental_coefficients, fundamental_features) + number_of_fundamental_features) / (2 * number_of_fundamental_features)  # this impact is in [0, 1]  (we are mapping from [-5, 5] to [0, 1])
                weights_total[stock_index] = (impact_of_fundamental * weights_fundamental[stock_index]) + weights_technical[stock_index]

        weights_total = np.array(weights_total)
        weights_total = weights_total / sum(weights_total)
        # ------ Calculating desired expected return and std using the obtained weights:
        expected_return_fuzzy = np.matrix((self.expected_returns)).T * np.matrix(weights_total).T
        std_fuzzy = np.sqrt(np.matrix(weights_total) * self.covariance_matrix * np.matrix(weights_total).T)

        if (include_technical_analysis is True) and (include_fundamental_analysis is False):
            weights_fundamental = []
        elif (include_technical_analysis is False) and (include_fundamental_analysis is True):
            weights_technical = []
            weights_technical_matrix = []
        elif (include_technical_analysis is False) and (include_fundamental_analysis is False):
            print('Error: at least one of the include_technical_analysis and include_fundamental_analysis should be True...')
            return -1

        return weights_total, weights_technical, weights_technical_matrix, weights_fundamental, expected_return_fuzzy, std_fuzzy

    def do_Fuzzy_logic_fundamental(self, fundamental_range, w_range, fundamental_coefficient_range,
                                   fundamental_features, fundamental_coefficients):
        # generate membership functions:
        fundamental_minus_high_MembershipFunction, fundamental_high_MembershipFunction, \
        fundamental_coefficient_minus_high_MembershipFunction, fundamental_coefficient_minus_medium_MembershipFunction, \
        fundamental_coefficient_medium_MembershipFunction, fundamental_coefficient_high_MembershipFunction, \
        w_low_MembershipFunction, w_medium_MembershipFunction, w_high_MembershipFunction = \
            self.generate_fuzzy_membership_functions_fundamental(fundamental_range=fundamental_range, w_range=w_range,
                                                                 fundamental_coefficient_range=fundamental_coefficient_range)
        # fuzzification:
        ######### AR:
        crisp_value = fundamental_features[0]
        AR_minus_high_FUZZIFIED = self.fuzzification(membership_function=fundamental_minus_high_MembershipFunction,
                                                          range_crisp_value=fundamental_range, crisp_value=crisp_value)
        crisp_value = fundamental_features[0]
        AR_high_FUZZIFIED = self.fuzzification(membership_function=fundamental_high_MembershipFunction,
                                                     range_crisp_value=fundamental_range, crisp_value=crisp_value)
        ######### CAPX:
        crisp_value = fundamental_features[1]
        CAPX_minus_high_FUZZIFIED = self.fuzzification(membership_function=fundamental_minus_high_MembershipFunction,
                                                     range_crisp_value=fundamental_range, crisp_value=crisp_value)
        crisp_value = fundamental_features[1]
        CAPX_high_FUZZIFIED = self.fuzzification(membership_function=fundamental_high_MembershipFunction,
                                               range_crisp_value=fundamental_range, crisp_value=crisp_value)
        ######### INV:
        crisp_value = fundamental_features[2]
        INV_minus_high_FUZZIFIED = self.fuzzification(membership_function=fundamental_minus_high_MembershipFunction,
                                                     range_crisp_value=fundamental_range, crisp_value=crisp_value)
        crisp_value = fundamental_features[2]
        INV_high_FUZZIFIED = self.fuzzification(membership_function=fundamental_high_MembershipFunction,
                                               range_crisp_value=fundamental_range, crisp_value=crisp_value)
        ######### GM:
        crisp_value = fundamental_features[3]
        GM_minus_high_FUZZIFIED = self.fuzzification(membership_function=fundamental_minus_high_MembershipFunction,
                                                     range_crisp_value=fundamental_range, crisp_value=crisp_value)
        crisp_value = fundamental_features[3]
        GM_high_FUZZIFIED = self.fuzzification(membership_function=fundamental_high_MembershipFunction,
                                               range_crisp_value=fundamental_range, crisp_value=crisp_value)
        ######### ETR:
        crisp_value = fundamental_features[4]
        ETR_minus_high_FUZZIFIED = self.fuzzification(membership_function=fundamental_minus_high_MembershipFunction,
                                                     range_crisp_value=fundamental_range, crisp_value=crisp_value)
        crisp_value = fundamental_features[4]
        ETR_high_FUZZIFIED = self.fuzzification(membership_function=fundamental_high_MembershipFunction,
                                               range_crisp_value=fundamental_range, crisp_value=crisp_value)
        ######### fundamental coefficient of AR:
        crisp_value = fundamental_coefficients[0]
        coefficient_AR_minus_high_FUZZIFIED = self.fuzzification(membership_function=fundamental_coefficient_minus_high_MembershipFunction,
                                                      range_crisp_value=fundamental_coefficient_range, crisp_value=crisp_value)
        crisp_value = fundamental_coefficients[0]
        coefficient_AR_minus_medium_FUZZIFIED = self.fuzzification(membership_function=fundamental_coefficient_minus_medium_MembershipFunction,
                                                range_crisp_value=fundamental_coefficient_range, crisp_value=crisp_value)
        crisp_value = fundamental_coefficients[0]
        coefficient_AR_medium_FUZZIFIED = self.fuzzification(membership_function=fundamental_coefficient_medium_MembershipFunction,
                                                range_crisp_value=fundamental_coefficient_range, crisp_value=crisp_value)
        crisp_value = fundamental_coefficients[0]
        coefficient_AR_high_FUZZIFIED = self.fuzzification(membership_function=fundamental_coefficient_high_MembershipFunction,
                                                range_crisp_value=fundamental_coefficient_range, crisp_value=crisp_value)
        ######### fundamental coefficient of CAPX:
        crisp_value = fundamental_coefficients[1]
        coefficient_CAPX_minus_high_FUZZIFIED = self.fuzzification(
            membership_function=fundamental_coefficient_minus_high_MembershipFunction,
            range_crisp_value=fundamental_coefficient_range, crisp_value=crisp_value)
        crisp_value = fundamental_coefficients[1]
        coefficient_CAPX_minus_medium_FUZZIFIED = self.fuzzification(
            membership_function=fundamental_coefficient_minus_medium_MembershipFunction,
            range_crisp_value=fundamental_coefficient_range, crisp_value=crisp_value)
        crisp_value = fundamental_coefficients[1]
        coefficient_CAPX_medium_FUZZIFIED = self.fuzzification(
            membership_function=fundamental_coefficient_medium_MembershipFunction,
            range_crisp_value=fundamental_coefficient_range, crisp_value=crisp_value)
        crisp_value = fundamental_coefficients[1]
        coefficient_CAPX_high_FUZZIFIED = self.fuzzification(
            membership_function=fundamental_coefficient_high_MembershipFunction,
            range_crisp_value=fundamental_coefficient_range, crisp_value=crisp_value)
        ######### fundamental coefficient of INV:
        crisp_value = fundamental_coefficients[2]
        coefficient_INV_minus_high_FUZZIFIED = self.fuzzification(
            membership_function=fundamental_coefficient_minus_high_MembershipFunction,
            range_crisp_value=fundamental_coefficient_range, crisp_value=crisp_value)
        crisp_value = fundamental_coefficients[2]
        coefficient_INV_minus_medium_FUZZIFIED = self.fuzzification(
            membership_function=fundamental_coefficient_minus_medium_MembershipFunction,
            range_crisp_value=fundamental_coefficient_range, crisp_value=crisp_value)
        crisp_value = fundamental_coefficients[2]
        coefficient_INV_medium_FUZZIFIED = self.fuzzification(
            membership_function=fundamental_coefficient_medium_MembershipFunction,
            range_crisp_value=fundamental_coefficient_range, crisp_value=crisp_value)
        crisp_value = fundamental_coefficients[2]
        coefficient_INV_high_FUZZIFIED = self.fuzzification(
            membership_function=fundamental_coefficient_high_MembershipFunction,
            range_crisp_value=fundamental_coefficient_range, crisp_value=crisp_value)
        ######### fundamental coefficient of GM:
        crisp_value = fundamental_coefficients[3]
        coefficient_GM_minus_high_FUZZIFIED = self.fuzzification(
            membership_function=fundamental_coefficient_minus_high_MembershipFunction,
            range_crisp_value=fundamental_coefficient_range, crisp_value=crisp_value)
        crisp_value = fundamental_coefficients[3]
        coefficient_GM_minus_medium_FUZZIFIED = self.fuzzification(
            membership_function=fundamental_coefficient_minus_medium_MembershipFunction,
            range_crisp_value=fundamental_coefficient_range, crisp_value=crisp_value)
        crisp_value = fundamental_coefficients[3]
        coefficient_GM_medium_FUZZIFIED = self.fuzzification(
            membership_function=fundamental_coefficient_medium_MembershipFunction,
            range_crisp_value=fundamental_coefficient_range, crisp_value=crisp_value)
        crisp_value = fundamental_coefficients[3]
        coefficient_GM_high_FUZZIFIED = self.fuzzification(
            membership_function=fundamental_coefficient_high_MembershipFunction,
            range_crisp_value=fundamental_coefficient_range, crisp_value=crisp_value)
        ######### fundamental coefficient of ETR:
        crisp_value = fundamental_coefficients[4]
        coefficient_ETR_minus_high_FUZZIFIED = self.fuzzification(
            membership_function=fundamental_coefficient_minus_high_MembershipFunction,
            range_crisp_value=fundamental_coefficient_range, crisp_value=crisp_value)
        crisp_value = fundamental_coefficients[4]
        coefficient_ETR_minus_medium_FUZZIFIED = self.fuzzification(
            membership_function=fundamental_coefficient_minus_medium_MembershipFunction,
            range_crisp_value=fundamental_coefficient_range, crisp_value=crisp_value)
        crisp_value = fundamental_coefficients[4]
        coefficient_ETR_medium_FUZZIFIED = self.fuzzification(
            membership_function=fundamental_coefficient_medium_MembershipFunction,
            range_crisp_value=fundamental_coefficient_range, crisp_value=crisp_value)
        crisp_value = fundamental_coefficients[4]
        coefficient_ETR_high_FUZZIFIED = self.fuzzification(
            membership_function=fundamental_coefficient_high_MembershipFunction,
            range_crisp_value=fundamental_coefficient_range, crisp_value=crisp_value)

        aggregated_rules_fundamental = self.fuzzy_rules_fundamental(AR_minus_high_FUZZIFIED, AR_high_FUZZIFIED,
                                CAPX_minus_high_FUZZIFIED, CAPX_high_FUZZIFIED,
                                INV_minus_high_FUZZIFIED, INV_high_FUZZIFIED,
                                GM_minus_high_FUZZIFIED, GM_high_FUZZIFIED,
                                ETR_minus_high_FUZZIFIED, ETR_high_FUZZIFIED,
                                coefficient_AR_minus_high_FUZZIFIED, coefficient_AR_minus_medium_FUZZIFIED,
                                coefficient_AR_medium_FUZZIFIED, coefficient_AR_high_FUZZIFIED,
                                coefficient_CAPX_minus_high_FUZZIFIED, coefficient_CAPX_minus_medium_FUZZIFIED,
                                coefficient_CAPX_medium_FUZZIFIED, coefficient_CAPX_high_FUZZIFIED,
                                coefficient_INV_minus_high_FUZZIFIED, coefficient_INV_minus_medium_FUZZIFIED,
                                coefficient_INV_medium_FUZZIFIED, coefficient_INV_high_FUZZIFIED,
                                coefficient_GM_minus_high_FUZZIFIED, coefficient_GM_minus_medium_FUZZIFIED,
                                coefficient_GM_medium_FUZZIFIED, coefficient_GM_high_FUZZIFIED,
                                coefficient_ETR_minus_high_FUZZIFIED, coefficient_ETR_minus_medium_FUZZIFIED,
                                coefficient_ETR_medium_FUZZIFIED, coefficient_ETR_high_FUZZIFIED,
                                w_low_MembershipFunction, w_medium_MembershipFunction, w_high_MembershipFunction)

        ################## defuzzification:
        weight = self.defuzzification(aggregated_rules=aggregated_rules_fundamental, range_crisp_value=w_range)
        return weight

    def generate_fuzzy_membership_functions_fundamental(self, fundamental_range, w_range, fundamental_coefficient_range):
        ##### input:
        # membership functions of fundamentals:
        fundamental_minus_high_MembershipFunction = fuzz.trimf(fundamental_range, [-1, -1, 0.33])
        fundamental_high_MembershipFunction = fuzz.trimf(fundamental_range, [-0.33, 1, 1])
        # membership functions of fundamental coefficient:
        fundamental_coefficient_minus_high_MembershipFunction = fuzz.trimf(fundamental_range, [-1, -1, 0])
        fundamental_coefficient_minus_medium_MembershipFunction = fuzz.trimf(fundamental_range, [-1, -0.5, 0])
        fundamental_coefficient_medium_MembershipFunction = fuzz.trimf(fundamental_range, [0, 0.5, 1])
        fundamental_coefficient_high_MembershipFunction = fuzz.trimf(fundamental_range, [0, 1, 1])
        ##### output:
        # membership functions of weights:
        w_low_MembershipFunction = fuzz.trimf(w_range, [0, 0, 0.66])
        w_medium_MembershipFunction = fuzz.trimf(w_range, [0.33, 0.5, 0.66])
        w_high_MembershipFunction = fuzz.trimf(w_range, [0.33, 1, 1])
        return fundamental_minus_high_MembershipFunction, fundamental_high_MembershipFunction, \
               fundamental_coefficient_minus_high_MembershipFunction, fundamental_coefficient_minus_medium_MembershipFunction, \
               fundamental_coefficient_medium_MembershipFunction, fundamental_coefficient_high_MembershipFunction, \
               w_low_MembershipFunction, w_medium_MembershipFunction, w_high_MembershipFunction

    def fuzzy_rules_fundamental(self,AR_minus_high_FUZZIFIED, AR_high_FUZZIFIED,
               CAPX_minus_high_FUZZIFIED, CAPX_high_FUZZIFIED,
               INV_minus_high_FUZZIFIED, INV_high_FUZZIFIED,
               GM_minus_high_FUZZIFIED, GM_high_FUZZIFIED,
               ETR_minus_high_FUZZIFIED, ETR_high_FUZZIFIED,
               coefficient_AR_minus_high_FUZZIFIED, coefficient_AR_minus_medium_FUZZIFIED,
               coefficient_AR_medium_FUZZIFIED, coefficient_AR_high_FUZZIFIED,
               coefficient_CAPX_minus_high_FUZZIFIED, coefficient_CAPX_minus_medium_FUZZIFIED,
               coefficient_CAPX_medium_FUZZIFIED, coefficient_CAPX_high_FUZZIFIED,
               coefficient_INV_minus_high_FUZZIFIED, coefficient_INV_minus_medium_FUZZIFIED,
               coefficient_INV_medium_FUZZIFIED, coefficient_INV_high_FUZZIFIED,
               coefficient_GM_minus_high_FUZZIFIED, coefficient_GM_minus_medium_FUZZIFIED,
               coefficient_GM_medium_FUZZIFIED, coefficient_GM_high_FUZZIFIED,
               coefficient_ETR_minus_high_FUZZIFIED, coefficient_ETR_minus_medium_FUZZIFIED,
               coefficient_ETR_medium_FUZZIFIED, coefficient_ETR_high_FUZZIFIED,
               w_low_MembershipFunction, w_medium_MembershipFunction, w_high_MembershipFunction):
        rules = []
          # considering merely own stock
        number_of_rules = 40
        k = 0
        for rule_index in range(number_of_rules):
            if rule_index == k:
                if_rule = coefficient_AR_high_FUZZIFIED
                if_rule = np.fmin(if_rule, AR_high_FUZZIFIED)
                if_rule = np.fmin(if_rule, w_high_MembershipFunction)
            elif rule_index == k+1:
                if_rule = coefficient_AR_high_FUZZIFIED
                if_rule = np.fmin(if_rule, AR_minus_high_FUZZIFIED)
                if_rule = np.fmin(if_rule, w_low_MembershipFunction)
            elif rule_index == k+2:
                if_rule = coefficient_AR_medium_FUZZIFIED
                if_rule = np.fmin(if_rule, AR_high_FUZZIFIED)
                if_rule = np.fmin(if_rule, w_medium_MembershipFunction)
            elif rule_index == k+3:
                if_rule = coefficient_AR_medium_FUZZIFIED
                if_rule = np.fmin(if_rule, AR_minus_high_FUZZIFIED)
                if_rule = np.fmin(if_rule, w_low_MembershipFunction)
            elif rule_index == k+4:
                if_rule = coefficient_AR_minus_medium_FUZZIFIED
                if_rule = np.fmin(if_rule, AR_high_FUZZIFIED)
                if_rule = np.fmin(if_rule, w_low_MembershipFunction)
            elif rule_index == k+5:
                if_rule = coefficient_AR_minus_medium_FUZZIFIED
                if_rule = np.fmin(if_rule, AR_minus_high_FUZZIFIED)
                if_rule = np.fmin(if_rule, w_medium_MembershipFunction)
            elif rule_index == k+6:
                if_rule = coefficient_AR_minus_high_FUZZIFIED
                if_rule = np.fmin(if_rule, AR_high_FUZZIFIED)
                if_rule = np.fmin(if_rule, w_low_MembershipFunction)
            elif rule_index == k+7:
                if_rule = coefficient_AR_minus_high_FUZZIFIED
                if_rule = np.fmin(if_rule, AR_minus_high_FUZZIFIED)
                if_rule = np.fmin(if_rule, w_high_MembershipFunction)

            if rule_index == k+8:
                if_rule = coefficient_CAPX_high_FUZZIFIED
                if_rule = np.fmin(if_rule, CAPX_high_FUZZIFIED)
                if_rule = np.fmin(if_rule, w_high_MembershipFunction)
            elif rule_index == k + 9:
                if_rule = coefficient_CAPX_high_FUZZIFIED
                if_rule = np.fmin(if_rule, CAPX_minus_high_FUZZIFIED)
                if_rule = np.fmin(if_rule, w_low_MembershipFunction)
            elif rule_index == k + 10:
                if_rule = coefficient_CAPX_medium_FUZZIFIED
                if_rule = np.fmin(if_rule, CAPX_high_FUZZIFIED)
                if_rule = np.fmin(if_rule, w_medium_MembershipFunction)
            elif rule_index == k + 11:
                if_rule = coefficient_CAPX_medium_FUZZIFIED
                if_rule = np.fmin(if_rule, CAPX_minus_high_FUZZIFIED)
                if_rule = np.fmin(if_rule, w_low_MembershipFunction)
            elif rule_index == k + 12:
                if_rule = coefficient_CAPX_minus_medium_FUZZIFIED
                if_rule = np.fmin(if_rule, CAPX_high_FUZZIFIED)
                if_rule = np.fmin(if_rule, w_low_MembershipFunction)
            elif rule_index == k + 13:
                if_rule = coefficient_CAPX_minus_medium_FUZZIFIED
                if_rule = np.fmin(if_rule, CAPX_minus_high_FUZZIFIED)
                if_rule = np.fmin(if_rule, w_medium_MembershipFunction)
            elif rule_index == k + 14:
                if_rule = coefficient_CAPX_minus_high_FUZZIFIED
                if_rule = np.fmin(if_rule, CAPX_high_FUZZIFIED)
                if_rule = np.fmin(if_rule, w_low_MembershipFunction)
            elif rule_index == k + 15:
                if_rule = coefficient_CAPX_minus_high_FUZZIFIED
                if_rule = np.fmin(if_rule, CAPX_minus_high_FUZZIFIED)
                if_rule = np.fmin(if_rule, w_high_MembershipFunction)

            if rule_index == k + 16:
                if_rule = coefficient_INV_high_FUZZIFIED
                if_rule = np.fmin(if_rule, INV_high_FUZZIFIED)
                if_rule = np.fmin(if_rule, w_high_MembershipFunction)
            elif rule_index == k + 17:
                if_rule = coefficient_INV_high_FUZZIFIED
                if_rule = np.fmin(if_rule, INV_minus_high_FUZZIFIED)
                if_rule = np.fmin(if_rule, w_low_MembershipFunction)
            elif rule_index == k + 18:
                if_rule = coefficient_INV_medium_FUZZIFIED
                if_rule = np.fmin(if_rule, INV_high_FUZZIFIED)
                if_rule = np.fmin(if_rule, w_medium_MembershipFunction)
            elif rule_index == k + 19:
                if_rule = coefficient_INV_medium_FUZZIFIED
                if_rule = np.fmin(if_rule, INV_minus_high_FUZZIFIED)
                if_rule = np.fmin(if_rule, w_low_MembershipFunction)
            elif rule_index == k + 20:
                if_rule = coefficient_INV_minus_medium_FUZZIFIED
                if_rule = np.fmin(if_rule, INV_high_FUZZIFIED)
                if_rule = np.fmin(if_rule, w_low_MembershipFunction)
            elif rule_index == k + 21:
                if_rule = coefficient_INV_minus_medium_FUZZIFIED
                if_rule = np.fmin(if_rule, INV_minus_high_FUZZIFIED)
                if_rule = np.fmin(if_rule, w_medium_MembershipFunction)
            elif rule_index == k + 22:
                if_rule = coefficient_INV_minus_high_FUZZIFIED
                if_rule = np.fmin(if_rule, INV_high_FUZZIFIED)
                if_rule = np.fmin(if_rule, w_low_MembershipFunction)
            elif rule_index == k + 23:
                if_rule = coefficient_INV_minus_high_FUZZIFIED
                if_rule = np.fmin(if_rule, INV_minus_high_FUZZIFIED)
                if_rule = np.fmin(if_rule, w_high_MembershipFunction)

            if rule_index == k + 24:
                if_rule = coefficient_GM_high_FUZZIFIED
                if_rule = np.fmin(if_rule, GM_high_FUZZIFIED)
                if_rule = np.fmin(if_rule, w_high_MembershipFunction)
            elif rule_index == k + 25:
                if_rule = coefficient_GM_high_FUZZIFIED
                if_rule = np.fmin(if_rule, GM_minus_high_FUZZIFIED)
                if_rule = np.fmin(if_rule, w_low_MembershipFunction)
            elif rule_index == k + 26:
                if_rule = coefficient_GM_medium_FUZZIFIED
                if_rule = np.fmin(if_rule, GM_high_FUZZIFIED)
                if_rule = np.fmin(if_rule, w_medium_MembershipFunction)
            elif rule_index == k + 27:
                if_rule = coefficient_GM_medium_FUZZIFIED
                if_rule = np.fmin(if_rule, GM_minus_high_FUZZIFIED)
                if_rule = np.fmin(if_rule, w_low_MembershipFunction)
            elif rule_index == k + 28:
                if_rule = coefficient_GM_minus_medium_FUZZIFIED
                if_rule = np.fmin(if_rule, GM_high_FUZZIFIED)
                if_rule = np.fmin(if_rule, w_low_MembershipFunction)
            elif rule_index == k + 29:
                if_rule = coefficient_GM_minus_medium_FUZZIFIED
                if_rule = np.fmin(if_rule, GM_minus_high_FUZZIFIED)
                if_rule = np.fmin(if_rule, w_medium_MembershipFunction)
            elif rule_index == k + 30:
                if_rule = coefficient_GM_minus_high_FUZZIFIED
                if_rule = np.fmin(if_rule, GM_high_FUZZIFIED)
                if_rule = np.fmin(if_rule, w_low_MembershipFunction)
            elif rule_index == k + 31:
                if_rule = coefficient_GM_minus_high_FUZZIFIED
                if_rule = np.fmin(if_rule, GM_minus_high_FUZZIFIED)
                if_rule = np.fmin(if_rule, w_high_MembershipFunction)

            if rule_index == k + 32:
                if_rule = coefficient_ETR_high_FUZZIFIED
                if_rule = np.fmin(if_rule, ETR_high_FUZZIFIED)
                if_rule = np.fmin(if_rule, w_high_MembershipFunction)
            elif rule_index == k + 33:
                if_rule = coefficient_ETR_high_FUZZIFIED
                if_rule = np.fmin(if_rule, ETR_minus_high_FUZZIFIED)
                if_rule = np.fmin(if_rule, w_low_MembershipFunction)
            elif rule_index == k + 34:
                if_rule = coefficient_ETR_medium_FUZZIFIED
                if_rule = np.fmin(if_rule, ETR_high_FUZZIFIED)
                if_rule = np.fmin(if_rule, w_medium_MembershipFunction)
            elif rule_index == k + 35:
                if_rule = coefficient_ETR_medium_FUZZIFIED
                if_rule = np.fmin(if_rule, ETR_minus_high_FUZZIFIED)
                if_rule = np.fmin(if_rule, w_low_MembershipFunction)
            elif rule_index == k + 36:
                if_rule = coefficient_ETR_minus_medium_FUZZIFIED
                if_rule = np.fmin(if_rule, ETR_high_FUZZIFIED)
                if_rule = np.fmin(if_rule, w_low_MembershipFunction)
            elif rule_index == k + 37:
                if_rule = coefficient_ETR_minus_medium_FUZZIFIED
                if_rule = np.fmin(if_rule, ETR_minus_high_FUZZIFIED)
                if_rule = np.fmin(if_rule, w_medium_MembershipFunction)
            elif rule_index == k + 38:
                if_rule = coefficient_ETR_minus_high_FUZZIFIED
                if_rule = np.fmin(if_rule, ETR_high_FUZZIFIED)
                if_rule = np.fmin(if_rule, w_low_MembershipFunction)
            elif rule_index == k + 39:
                if_rule = coefficient_ETR_minus_high_FUZZIFIED
                if_rule = np.fmin(if_rule, ETR_minus_high_FUZZIFIED)
                if_rule = np.fmin(if_rule, w_high_MembershipFunction)

            rules.append(if_rule)

        aggregated_two_rules = rules[0]
        for rule_index in range(1, number_of_rules):
            rule = rules[rule_index]
            aggregated_two_rules = np.fmax(aggregated_two_rules, rule)
        aggregated_rules = aggregated_two_rules
        return aggregated_rules  # note: aggregated_rules is the fuzzified output (which needs to be defuzzified)