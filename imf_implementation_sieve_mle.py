from sklearn.neighbors import KernelDensity
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy
import warnings
import inspect
import os
import glob
from mle_optimizer import *
import argparse
import pickle
from imf_implementation_helper import *


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Arguments for optimizer.')
    parser.add_argument("--method", type=str, help="optimization method.")
    parser.add_argument("--max_iters", type=int, help="maximum number of iterations.")
    parser.add_argument("--gtol", type=float, help="gradient tolerance.")
    parser.add_argument("--num_workers", type=int, help="number of workers.")
    parser.add_argument("--xtol", type=float, help="parameter tolerance for convergence.")
    parser.add_argument("--ftol", type=float, help="function value tolerance for convergence.")
    parser.add_argument("--resume", type=float, help="resume from last saved params.")
    parser.add_argument("--viirs_data", type=int, help="1 if optimization is to be done on VIIRS data(2013-2021), 0 if on DMSP-OLS data(1992-2012)")

    args = parser.parse_args()

    if args.viirs_data == 1:
        years = [year for year in range(2013, 2021+1)]
    else:
        years = [year for year in range(1992, 2012+1)]

    print("Analysis will be done for the following years: ", years)

    ## Checking if the combined data file already exists
    csv_files = [file_.split("/")[-1] for file_ in glob.glob(os.getcwd()+"/data/*.csv")]
    file_to_check = "imf_data_combined_viirs.csv" if args.viirs_data == 1 else "imf_data_combined_dmsp_ols.csv"
    if file_to_check in csv_files:
        print("IMF data already present.")
        all_params_df = pd.read_csv(os.getcwd()+"/data/{}".format(file_to_check))
    else:
        ## Combining NTL data for all the years
        print("Combining data for all the years.")
        pop_df = pd.read_csv(os.getcwd()+"/data/population/API_SP.POP.TOTL_DS2_en_csv_v2_5358404.csv")
        combined_ntl_df = get_all_ntl_countries_data(years, pop_df, annual=True)

        ## Reading and processing SPI and Centroid data.
        spi_centroid_filename = os.getcwd()+"/data/SPI_index_mean_and_centroid_latitude.csv"
        spi_and_centroid_df = get_spi_and_centroid_df(spi_centroid_filename)

        ## Reading and processing GDP per capita data.
        gdp_filename = os.getcwd()+"/data/gdp/gdp_per_capita.csv"
        gdp_df = get_gdp_df(gdp_filename, years)

        ## Combining all data
        all_params_df = merge_all_data(combined_ntl_df, spi_and_centroid_df, gdp_df, args.viirs_data)

    method = args.method

    if args.viirs_data == 1:
        num_hermite_polynomials = 4
    else:
        num_hermite_polynomials = 6

    max_iters = args.max_iters
    init_params = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    # init_params = [0.5, 1.0, -0.1, 1.0, 1.0, 1.0, 1.0, 1.0]
    # init_params = [ 0.13145964, -1.22748102, 0.26903948, 4.55958534, 2.47690282, 2.02219325, 1.09059137, 1.75088386]
    bounds = ((None, None), (None, None), (None, None), (0.0, 10**2), (0.0, 10**2), (0.0, 10**2), 
            (0.0, 10**2), (0.0, 10**2))

    if method == "L-BFGS-B":
        method_options = {
            "maxfun": max_iters,
            "iprint": 1,
            "gtol": args.gtol
        }
        if args.resume == 1:
            init_params = [0.81250852, 0.82166194, 0.50729731, 1.28561861, 1.36433879, 1.22308101, 1.08774401, 1.01040638]
    elif method == "Newton-CG":
        method_options = {
            "maxiter": max_iters,
            "xtol": args.xtol,
            "return_all": True
        }
        bounds=None
    elif method == "Nelder-Mead":
        method_options = {
            "maxiter": max_iters,
            "xatol": args.xtol,
            "fatol": args.ftol,
            "adaptive": True,
            "return_all": True
        }
        if args.resume == 1:
            if args.viirs_data == 1:
                init_params = [0.47917767, 0.44044462, -0.01744728, 2.23039898, 1.57816555, 1.69046568, 0.90175838, 0.99124738]# after 670 iterations
            else:
                init_params = [0.81290302, 0.52788103, 0.06277217, 1.96739031, 1.41530485, 1.37860406, 1.78302053, 1.72199836] #after 400 iterations
                # init_params = [0.86107254, 0.58993523, 0.22551581, 1.67886088, 1.40225668, 1.44557908, 1.38426667, 1.24810972] #after 300 iterations
                # init_params = [0.84972755, 0.61798574, 0.32911206, 1.56106994, 1.36283057, 1.39878749, 1.22980568, 1.12849776]# after 200 iterations
            # init_params = [0.41747294, 0.26844452, 0.02256721, 4.53437794, 2.55153569, 2.19104476, 1.4223711, 1.1841374 ]# after 550 iterations when starting from [1, 1, 1 .....]
            # init_params = [0.41214143, 0.27077883, 0.02194174, 4.53786928, 2.58460825, 2.16061576, 1.41182132, 1.17503154] # After 527 iterations when starting from [1, 1, 1 .....]
            # init_params = [ 0.57989516, 0.6719327, -0.05119052, 1.5737528, 1.71449321, 1.53676772, 1.06031199, 1.07512463] # after 182 iterations when starting from [1, 1, 1 .....]
    else:
        method_options = {}

    optimizer = Optimizer(all_params_df, years, num_hermite_polynomials, init_params, max_iters, bounds=bounds, 
                    method=method, method_options=method_options, num_workers=args.num_workers)

    mle_model = optimizer.optimize()

    for key, val in mle_model.items():
        print(key, val)

    save_filename = "mle_params_viirs.pkl" if args.viirs_data == 1 else "mle_params_dmsp_ols.pkl"
    with open("./data/"+save_filename, 'wb') as f:
        pickle.dump(mle_model, f)

    print("params saved.")




