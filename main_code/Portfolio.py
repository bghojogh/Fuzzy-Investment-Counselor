import numpy as np

# inspired by: https://blog.quantopian.com/markowitz-portfolio-optimization-2/

class Portfolio:

    def __init__(self, covariance_matrix, expected_returns, desired_variance, risk_tolerance_percentage):
        self.covariance_matrix = covariance_matrix
        self.expected_returns = expected_returns
        self.desired_variance = desired_variance
        self.number_of_stocks = len(expected_returns)
        self.risk_tolerance_percentage = risk_tolerance_percentage

    def random_weights(self, number_of_weights):
        w = np.random.rand(number_of_weights)
        # w = 2 * (np.random.rand(number_of_weights)) - 1
        w = w / sum(w)
        return w

    def random_portfolio_points(self):
        w = self.random_weights(number_of_weights=self.number_of_stocks)
        w = np.matrix(w).T

        expected_return_portfilio_points = np.matrix((self.expected_returns)).T * w
        std_portfolio_points = np.sqrt(w.T * self.covariance_matrix * w)

        expected_return_portfilio_points = float(expected_return_portfilio_points)
        std_portfolio_points = float(std_portfolio_points)
        w = np.array(w).ravel().tolist()  # changing matrix(vector) w to a list

        # This reduces outliers to keep plots pretty
        if std_portfolio_points > 2:
            return self.random_portfolio_points()
        return expected_return_portfilio_points, std_portfolio_points, w

    def find_the_desired_optimum_portfolio(self, expected_return_portfolio_points_list, std_portfolio_points_list, weights_list):
        epsilon = 0.01 * (max(std_portfolio_points_list) - min(std_portfolio_points_list))
        number_of_portfolio_points = len(expected_return_portfolio_points_list)
        risk_tolerance = min(std_portfolio_points_list) + ((self.risk_tolerance_percentage / 100) * (max(std_portfolio_points_list) - min(std_portfolio_points_list)))
        need_more_search = True
        while need_more_search:  # to check whether there exists any point in the region close to risk tolerance
            portfolio_optimum_expected_return = -1 * np.inf
            for point_index in range(number_of_portfolio_points):
                if abs(std_portfolio_points_list[point_index] - (risk_tolerance - 0.5*epsilon)) <= epsilon/2:
                    # the point has the risk close to risk tolerance (between tolerance-epsilon and tolerance)
                    if expected_return_portfolio_points_list[point_index] > portfolio_optimum_expected_return:
                        portfolio_optimum_expected_return = expected_return_portfolio_points_list[point_index]
                        portfolio_optimum_std = std_portfolio_points_list[point_index]
                        portfolio_optimum_weights = weights_list[point_index]
                        need_more_search = False
            epsilon += 0.001

        portfolio_optimum_expected_return_list, portfolio_optimum_std_list, portfolio_optimum_weights_list = [], [], []
        for point_index in range(number_of_portfolio_points):
            if (expected_return_portfolio_points_list[point_index] > portfolio_optimum_expected_return) and \
               (std_portfolio_points_list[point_index] < portfolio_optimum_std):
                portfolio_optimum_expected_return_temp = expected_return_portfolio_points_list[point_index]
                portfolio_optimum_std_temp = std_portfolio_points_list[point_index]
                portfolio_optimum_weights_temp = weights_list[point_index]
                portfolio_optimum_expected_return_list.append(portfolio_optimum_expected_return_temp)
                portfolio_optimum_std_list.append(portfolio_optimum_std_temp)
                portfolio_optimum_weights_list.append(portfolio_optimum_weights_temp)
        if not (not portfolio_optimum_expected_return_list): # If aa is not empty (If we do not have already the optimum point)
            optimum_index = np.argmax(np.array(portfolio_optimum_expected_return_list))
            portfolio_optimum_expected_return = portfolio_optimum_expected_return_list[optimum_index]
            portfolio_optimum_std = portfolio_optimum_std_list[optimum_index]
            portfolio_optimum_weights = portfolio_optimum_weights_list[optimum_index]
        return portfolio_optimum_expected_return, portfolio_optimum_std, portfolio_optimum_weights, risk_tolerance