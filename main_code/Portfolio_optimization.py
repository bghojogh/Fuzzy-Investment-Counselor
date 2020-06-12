from math import sqrt
from cvxopt import matrix
from cvxopt.blas import dot
from cvxopt.solvers import qp, options


# inspired by: https://blog.quantopian.com/markowitz-portfolio-optimization-2/

class Portfolio_optimization:

    def __init__(self, covariance_matrix, expected_returns, desired_variance, risk_tolerance_percentage):
        self.covariance_matrix = covariance_matrix
        self.expected_returns = expected_returns
        self.desired_variance = desired_variance
        self.number_of_stocks = len(expected_returns)
        self.risk_tolerance_percentage = risk_tolerance_percentage

    def find_the_desired_optimum_portfolio(self):
        n = 4
        S = matrix(self.covariance_matrix)
        print(S)
        xxx
        pbar = matrix([.12, .10, .07, .03])
        
        G = matrix(0.0, (n,n))
        G[::n+1] = -1.0
        h = matrix(0.0, (n,1))
        A = matrix(1.0, (1,n))
        b = matrix(1.0)
        
        N = 100
        mus = [ 10**(5.0*t/N-1.0) for t in range(N) ]
        options['show_progress'] = False
        xs = [ qp(mu*S, -pbar, G, h, A, b)['x'] for mu in mus ]
        returns = [ dot(pbar,x) for x in xs ]
        risks = [ sqrt(dot(x, S*x)) for x in xs ]
#        epsilon = 0.01 * (max(std_portfolio_points_list) - min(std_portfolio_points_list))
#        number_of_portfolio_points = len(expected_return_portfolio_points_list)
#        risk_tolerance = min(std_portfolio_points_list) + ((self.risk_tolerance_percentage / 100) * (max(std_portfolio_points_list) - min(std_portfolio_points_list)))
#        need_more_search = True
#        while need_more_search:  # to check whether there exists any point in the region close to risk tolerance
#            portfolio_optimum_expected_return = -1 * np.inf
#            for point_index in range(number_of_portfolio_points):
#                if abs(std_portfolio_points_list[point_index] - (risk_tolerance - 0.5*epsilon)) <= epsilon/2:
#                    # the point has the risk close to risk tolerance (between tolerance-epsilon and tolerance)
#                    if expected_return_portfolio_points_list[point_index] > portfolio_optimum_expected_return:
#                        portfolio_optimum_expected_return = expected_return_portfolio_points_list[point_index]
#                        portfolio_optimum_std = std_portfolio_points_list[point_index]
#                        portfolio_optimum_weights = weights_list[point_index]
#                        need_more_search = False
#            epsilon += 0.001
#
#        portfolio_optimum_expected_return_list, portfolio_optimum_std_list, portfolio_optimum_weights_list = [], [], []
#        for point_index in range(number_of_portfolio_points):
#            if (expected_return_portfolio_points_list[point_index] > portfolio_optimum_expected_return) and \
#               (std_portfolio_points_list[point_index] < portfolio_optimum_std):
#                portfolio_optimum_expected_return_temp = expected_return_portfolio_points_list[point_index]
#                portfolio_optimum_std_temp = std_portfolio_points_list[point_index]
#                portfolio_optimum_weights_temp = weights_list[point_index]
#                portfolio_optimum_expected_return_list.append(portfolio_optimum_expected_return_temp)
#                portfolio_optimum_std_list.append(portfolio_optimum_std_temp)
#                portfolio_optimum_weights_list.append(portfolio_optimum_weights_temp)
#        if not (not portfolio_optimum_expected_return_list): # If aa is not empty (If we do not have already the optimum point)
#            optimum_index = np.argmax(np.array(portfolio_optimum_expected_return_list))
#            portfolio_optimum_expected_return = portfolio_optimum_expected_return_list[optimum_index]
#            portfolio_optimum_std = portfolio_optimum_std_list[optimum_index]
#            portfolio_optimum_weights = portfolio_optimum_weights_list[optimum_index]
        
        
        return portfolio_optimum_expected_return, portfolio_optimum_std, portfolio_optimum_weights, risk_tolerance
