import pandas as pd 
import glob
import argparse
import numpy as np
import pickle
from tqdm import tqdm
def calibrate_data(ntl_df, calibration_df, country_num_pixels, satellite_name, year):
    coeffs = calibration_df[(calibration_df["satellite"] == satellite_name) & (calibration_df["year"] == year)]
    c0, c1, c2 = coeffs.iloc[0][2:]

    ## Gets the overall minimum and maximum DN(digital number denoting light brightness) using the calibration coeffs.
    all_min_dn, all_max_dn = get_extreme_calibrated_values(calibration_df)

    all_calibrated_values = []
    all_avg_calibrated_values = []
    all_avg_ntl_values = []
    all_rescaled_calibrated_values = []
    all_avg_rescaled_calibrated_values = []

    for _, row in ntl_df.iterrows():
        region_code = row["Region Code"]
        num_pixels = country_num_pixels[region_code]
        if row["annual_ntl_sum"] is None or row["annual_ntl_sum_sq"] is None or num_pixels is None:
            calibrated_value, avg_calibrated_value, avg_ntl_value, rescaled_calibrated_value, avg_rescaled_calibrated_value = [None]*5
        else:
            calibrated_value = num_pixels*c0 + c1*row["annual_ntl_sum"] + c2*row["annual_ntl_sum_sq"]

            avg_ntl_value = row["annual_ntl_sum"]/num_pixels
            avg_calibrated_value = calibrated_value/num_pixels
            rescaled_calibrated_value = rescale_data(calibrated_value, all_min_dn, all_max_dn, num_pixels)
            avg_rescaled_calibrated_value = rescaled_calibrated_value/num_pixels

        all_calibrated_values.append(calibrated_value)
        all_avg_calibrated_values.append(avg_calibrated_value)
        all_avg_ntl_values.append(avg_ntl_value)
        all_rescaled_calibrated_values.append(rescaled_calibrated_value)
        all_avg_rescaled_calibrated_values.append(avg_rescaled_calibrated_value)

    ntl_df["calibrated_ntl_sum"] = all_calibrated_values
    ntl_df["avg_ntl"] = all_avg_ntl_values
    ntl_df["avg_calibrated"] = all_avg_calibrated_values
    ntl_df["rescaled_calibrated_ntl_sum"] = all_rescaled_calibrated_values
    ntl_df["avg_rescaled_calibrated"] = all_avg_rescaled_calibrated_values
    return ntl_df
        

def get_extreme_calibrated_values(coeffs_df):
    def quad_func(a, c0, c1, c2):
        return c0 + c1*a + c2*(a**2)

    vfunc = np.vectorize(quad_func)
    dn_values = np.arange(0, 64, 1)

    all_min, all_max = np.inf, -np.inf
    for _, row in coeffs_df.iterrows():
        c0, c1, c2 = row[2:]
        all_min = min(all_min, np.min(vfunc(dn_values, c0, c1, c2)))
        all_max = max(all_max, np.max(vfunc(dn_values, c0, c1, c2)))

    return all_min, all_max

def rescale_data(value, all_min, all_max, num_pixels):
    if num_pixels is None:
        return None
    return (value - all_min*num_pixels)*63.0/(all_max - all_min)

def combine_satellites_data(year, save):
    ## Combines different satellites' data for DMSP-OLS.
    for month_folder in glob.glob("./data/earth_observation_group/monthly/{}/*".format(year)):
        csv_files = [file for file in glob.glob(month_folder+"/*") if ".csv" in file]
        if len(csv_files) == 2:
            combined_df = None
            for file in csv_files:
                if combined_df is None:
                    combined_df = pd.read_csv(file)
                else:
                    new_df = pd.read_csv(file)
                    combined_df = combined_df.set_index("Region Code").join(new_df.set_index("Region Code"), 
                                                                            rsuffix="_1").reset_index()
                    combined_df["rescaled_calibrated_ntl_sum"] = combined_df[["rescaled_calibrated_ntl_sum", 
                                                                              "rescaled_calibrated_ntl_sum_1"]].mean(axis=1)

            cols_to_drop = list(combined_df.columns)
            cols_to_drop.remove("Region Code")
            cols_to_drop.remove("rescaled_calibrated_ntl_sum")
            combined_df.drop(columns=cols_to_drop, inplace=True)

            tmp = file.split("/")
            tmp[-1] = "countries_ntl_sum.csv"
            new_filename = '/'.join(tmp)
            if save == 1:
                combined_df.to_csv(new_filename, index=False)
            



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Arguments for NTL yearly aggregation')
    parser.add_argument("--years", nargs="+", type=int, help="years for which annual radiance is to be evaluated")
    parser.add_argument("--calibrate", required=False, type=int, help="1 to do the intercalibration.")
    parser.add_argument("--save_calibration", required=False, type=int, help="1 to save the calibrated data.")
    parser.add_argument("--combine_data", required=False, type=int, help="combine different satellites data.")
    parser.add_argument("--save_combined_data", required=False, type=int, help="save the combined data.")

    parser.print_help()
    print()
    print("_"*100)
    print("Remove this print statements after going through the code file and the arguments needed for it.")
    exit(0)

    args = parser.parse_args()
    
    years = args.years
    if years is None:
        years = np.arange(1992, 2012+1)

    ## Calibration of the monthly data
    if args.calibrate == 1:
        print("Calibrating data ...")
        calibration_df = pd.read_csv("./data/Elvidge_DMSP_intercalib_coef.csv")
        with open("./data/country_num_pixels.pkl", "rb") as f:
            country_num_pixels = pickle.load(f)

        monthly_basepath = "./data/earth_observation_group/monthly/"
        coeffs_filepath = "./data/Elvidge_DMSP_intercalib_coef.csv"

        for year in tqdm(years):
            print("year: ", year)
            for month_folder in glob.glob(monthly_basepath+str(year)+"/*"):
                for file in glob.glob(month_folder+"/*"):
                    if ".csv" in file:
                        ntl_df = pd.read_csv(file)
                        satellite_name = file.split("/")[-1].split(".")[0].split("_")[-1]
                        ntl_df = calibrate_data(ntl_df, calibration_df, country_num_pixels, satellite_name, year)

                        if args.save_calibration == 1:
                            ntl_df.to_csv(file, index=False)

    ## Combining different satellites data
    if args.combine_data == 1:
        print("Combining satellites data ...")
        for year in tqdm(years):
            print("year: ", year)
            combine_satellites_data(year, args.save_combined_data)
