## Maximum likelihood Optimizer

import numpy as np
import scipy as sc
from scipy.integrate import quad
from scipy.optimize import minimize, approx_fprime
from optimparallel import minimize_parallel

class Optimizer:
    def __init__(self, all_params_df, years, num_hermite_polynomials, init_params, max_iters, bounds=None, 
                method="BFGS", method_options=None, num_workers=1):
        self.all_params_df = all_params_df
        self.years = years
        self.num_hermite_polynomials = num_hermite_polynomials
        self.init_params = init_params
        self.bounds = bounds
        self.method = method
        self.method_options = method_options
        
        self.num_iters = 0
        self.max_iters = max_iters
        self.num_workers = num_workers
        
        # Probability of occurence of spi and centroid categories
        self.categories_prob = {
            "spi": {idx: row for idx, row in self.all_params_df["spi_category"].value_counts().items()},
            "centroid": {idx: row for idx, row in self.all_params_df["centroid_category"].value_counts().items()}
        }
        
    def callback(self, x):
        self.num_iters += 1
        
        print("Current params: ", x)
        print("_"*10)
        if self.num_iters >= self.max_iters:
            raise StopIteration
#             warnings.warn("Iterations limit reached, terminating optimization.", IterationLimit)
#             return True
        elif self.num_iters%3 == 0:
            print("Iterations elapsed: ", self.num_iters)
#             return False
        
        
    def integrand(self, y_star, y, gdp_error_std, z, ntl_error_std, regression_params, 
                  num_hermite_polynomials):
        ## Square in the following term is because f(y*|s_i) = f(y*|l_i) since we are using the same
        ## weights in both the cases
        real_gdp_pdf = self.get_real_gdp_pdf(y_star, num_hermite_polynomials)**2
        gdp_error_pdf = self.get_gdp_error_pdf(y_star, y, gdp_error_std)
        ntl_error_pdf = self.get_ntl_error_pdf(y_star, z, ntl_error_std, regression_params)

        return real_gdp_pdf * gdp_error_pdf * ntl_error_pdf
        
    ## returns log-likelihood for a single observation i.e single year NTL and GDP for a country.
    def get_log_llh_country(self, gdp, ntl, gdp_error_std, ntl_error_std, regression_params, spi_prob, 
                            centroid_prob, num_hermite_polynomials):
        gdp_log = np.log(gdp)
        ntl_log = np.log(ntl)

        integral = quad(self.integrand, 1, 6, args=(gdp_log, gdp_error_std, ntl_log, ntl_error_std, 
                                                    regression_params, num_hermite_polynomials))[0]

        res = np.log(integral) + np.log(spi_prob * centroid_prob)
        return res

    
    def get_real_gdp_pdf(self, y_star, num_hermite_polynomials):
        res = 0.0
        weight = 1/np.sqrt(num_hermite_polynomials) 
        for degree in range(1, num_hermite_polynomials+1):
            res += self.get_hermite_polynomial(y_star, degree) * weight
        return res**2


    def get_gdp_error_pdf(self, y_star, y, gdp_error_std):
        return sc.stats.norm.pdf(y - y_star, 0.0, gdp_error_std)

    def get_ntl_error_pdf(self, y_star, z, ntl_error_std, regression_params):
        z_pred = regression_params[0] + regression_params[1] * y_star + regression_params[2] * (y_star**2)
        return sc.stats.norm.pdf(z - z_pred, 0.0, ntl_error_std)


    def get_hermite_polynomial(self, x, degree):
        hermite_polynomial = {
            1: lambda x: x,
            2: lambda x: x**2 - 1,
            3: lambda x: x**3 - 3*x,
            4: lambda x: x**4 - 6*(x**2) + 3,
            5: lambda x: x**5 - 10*(x**3) + 15*x,
            6: lambda x: x**6 - 15*(x**4) + 45*(x**2) - 15
        }
        return hermite_polynomial[degree](x)
    
    
    def mle_norm(self, parameters):
        self.num_iters += 1
        theta_0, theta_1, theta_2, gdp_error_std_cat_1, gdp_error_std_cat_2, gdp_error_std_cat_3, ntl_error_std_cat_1, ntl_error_std_cat_2 = parameters

        error_terms_dict = {
            "spi": {
                1: gdp_error_std_cat_1,
                2: gdp_error_std_cat_2,
                3: gdp_error_std_cat_3
            },
            "centroid": {
                1: ntl_error_std_cat_1,
                2: ntl_error_std_cat_2
            }
        }

        ## Sum of joint loglikelihood for all the observations
        total_llhd = 0.0

        for _, row in self.all_params_df.iterrows():
            spi_cat = row["spi_category"]
            centroid_cat = row["centroid_category"]

            for year in self.years:
                gdp = row["gdp_{}".format(year)]
                ntl = row["annual_ntl_sum_per_capita_{}".format(year)]
                total_llhd += self.get_log_llh_country(gdp, ntl, error_terms_dict["spi"][spi_cat], 
                                                       error_terms_dict["centroid"][centroid_cat], 
                                                       (theta_0, theta_1, theta_2), 
                                                       self.categories_prob["spi"][spi_cat], 
                                                       self.categories_prob["centroid"][centroid_cat],
                                                       self.num_hermite_polynomials)
        print("llhd: ", total_llhd)
        print("params: ", parameters)
        
        return -1.0 * total_llhd
    
            
    def optimize(self):
        if self.method == "Newton-CG":
            jac = lambda x: approx_fprime(x, self.mle_norm, 0.01)
        else:
            jac = None
        mle_model = minimize(self.mle_norm, 
                             self.init_params, 
                             method=self.method,
                             jac=jac, 
                             callback=self.callback, 
                             options=self.method_options, 
                             bounds=self.bounds)
        # mle_model = minimize_parallel(
        #     fun=self.mle_norm, 
        #     x0=self.init_params, 
        #     options=self.method_options, 
        #     bounds=self.bounds, 
        #     parallel={
        #         "max_workers": self.num_workers, 
        #         "verbose": True})
        return mle_model