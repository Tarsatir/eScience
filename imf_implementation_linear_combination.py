from imf_implementation_sieve_mle import *
import numpy as np
import pandas as pd
import argparse
import os
import pickle

class LinearCombination:
    def __init__(self, years=[], dmsp_ols=True, countries=["BDI", "KEN", "RWA", "TZA","UGA"]):
        self.years = years
        if len(years) == 0:
            self.years = np.arange(1992, 2012+1) if dmsp_ols else np.arange(2013, 2021+1)

        self.dmsp_ols = dmsp_ols
        self.data_type = "dmsp_ols" if dmsp_ols else "viirs"
        self.countries = countries

        self.yearly_gdp_filepath = "./data/gdp/gdp_per_capita.csv" #Yearly per capita GDP filename
        self.monthly_gdp_filepath = "./data/gdp/monthly_gdp_per_capita.csv" #Monthly per capita GDP filename
        self.pop_filepath = "./data/population/API_SP.POP.TOTL_DS2_en_csv_v2_5358404.csv" #Population data filename
        self.spi_centroid_filename = "./data/SPI_index_mean_and_centroid_latitude.csv"

        self.all_params_filename = "./data/imf_data_combined_{}{}.csv".format(self.data_type, "_monthly")
        self.all_params_df = None
        self.all_models_filename = "./data/ntl_gdp_models.pkl"
        self.all_models = None
        self.revised_gdp_monthly_filename = "./data/revised_gdp_{}_monthly.csv".format(self.data_type)
        self.revised_gdp_df = None

        #Create monthly GDP file if it doesn't exist.
        if not os.path.exists(self.monthly_gdp_filepath):
            self.interpolate_gdp()

        #Create the combined data if it doesn't exist
        if not os.path.exists(self.all_params_filename):
            self.get_combined_data()
        else:
            self.all_params_df = pd.read_csv(self.all_params_filename)

        self.all_params_df.sort_values(by=["Region Code"], inplace=True)

        #Create all regression models for NTL and GDP
        if not os.path.exists(self.all_models_filename):
            self.get_ntl_gdp_model_all_settings()
        else:
            self.all_models = pickle.load(open(self.all_models_filename, "rb"))

        ##Reading the Sieve MLE coefficients
        self.mle_params = pickle.load(open("./data/mle_params_{}.pkl".format(self.data_type), "rb"))

        #Evaluate monthly gdp
        if not os.path.exists(self.revised_gdp_monthly_filename):
            self.evaluate_gdp_monthly()
        else:
            self.revised_gdp_df = pd.read_csv(self.revised_gdp_monthly_filename)

        
    def func_interp(fp, x, xp):
        return np.interp(x, xp, fp)


    def interpolate_gdp(self):
        if os.path.exist(self.monthly_gdp_filepath):
            return

        gdp_df = pd.read_csv(self.yearly_gdp_filepath)

        ## parameters for interpolation
        num_values = len(gdp_df.columns) - 2 
        x = list(np.arange(1, 12*(num_values-1)+1))
        xp = list(np.arange(0, 12*num_values, 12))

        interp_values = np.apply_along_axis(func_interp, 1, gdp_df.iloc[:, 2:], x, xp)

        interp_df = pd.DataFrame(gdp_df.iloc[:, :2], columns=["Country Name", "Country Code"])
        new_col_names = []
        for year in np.arange(1992, 2021+1):
            for month in range(1, 12+1):
                new_col_names.append("{}_{}".format(year, month))
        interp_df = pd.concat([interp_df, pd.DataFrame(interp_values, columns=new_col_names)], axis=1)
        interp_df.to_csv(self.monthly_gdp_filepath, index=False)


    def get_combined_data(self):
        pop_df = pd.read_csv(self.pop_filepath)
        combined_ntl_df = get_all_ntl_countries_data(self.years, pop_df, annual=False)
    
        ## Processing SPI and Centroid data.
        spi_and_centroid_df = get_spi_and_centroid_df(self.spi_centroid_filename)

        ## Processing GDP per capita data.
        gdp_df = get_gdp_df(self.monthly_gdp_filepath, self.years)

        ## Combining all data
        self.all_params_df = merge_all_data(combined_ntl_df, spi_and_centroid_df, gdp_df, is_viirs_data=not self.dmsp_ols, annual=False)


    def get_ntl_gdp_model_all_settings(self):
        all_models = {}
        for data_type in ["dmsp_ols", "viirs"]:
            for duration_type in ["monthly", "annual"]:
                df = pd.read_csv("./data/imf_data_combined_{}{}.csv".format(data_type, "_monthly" if duration_type=="monthly" else ""))
                X, y = create_one_hot_ntl_data(df, is_viirs_data=True if data_type=="viirs" else False, annual=True if duration_type=="annual" else False)
                model = get_ntl_gdp_model(X, y)
                all_models["{}_{}".format(data_type, duration_type)] = model

        self.all_models = all_models
        pickle.dump(all_models, open(self.all_models_filename, "wb"))


    def prepare_data_for_lambda_evaluation(self, annual=True):
        ##This function can also prepare the data in the same format for the monthly data. 

        ##Reading the all_params_df for annual data since the evaluation of lambda depends on the annual data.
        if annual:
            all_params_df = pd.read_csv("./data/imf_data_combined_{}.csv".format(self.data_type))
        else:
            all_params_df = pd.read_csv("./data/imf_data_combined_{}_monthly.csv".format(self.data_type))

        ##Creating a list of NTL columns so as to get the temporal data
        ntl_columns = [col for col in all_params_df.columns if "ntl" in col]
        if annual:
            ntl_columns = sorted(ntl_columns, key=lambda x: int(x.split("_")[-1]))
            temporal_dummy_ids = {"_".join(col.split("_")[-1:]): i for i, col in enumerate(ntl_columns)}
        else:
            ntl_columns = sorted(ntl_columns, key=lambda x: (int(x.split("_")[-2]), int(x.split("_")[-1])))
            temporal_dummy_ids = {"_".join(col.split("_")[-2:]): i for i, col in enumerate(ntl_columns)}
        

        ##Reading the MLE params
        spi_error_stds = self.mle_params["x"][3:6]
        cent_error_stds = self.mle_params["x"][6:]

        ##Getting the NTL-GDP coefficients
        if annual:
            reg_model = self.all_models["{}_annual".format(self.data_type)]
        else:
            reg_model = self.all_models["{}_monthly".format(self.data_type)]

        beta = reg_model.coef_[0][0]

        df_for_lambda = [] #id(country_name_year), y, z, a, y_hat, spi_error_std, cent_error_std

        for country_code in self.countries:
            country_info =  all_params_df[all_params_df["Region Code"] == country_code]
            country_id = country_info.index[0] + 1 # Added 1 because the first value in regression params is beta.
            coef_country_dummy = reg_model.coef_[0][country_id]
            spi_cat, cent_cat = country_info[["spi_category", "centroid_category"]].values[0]
            spi_error_std, cent_error_std = spi_error_stds[spi_cat-1], cent_error_stds[cent_cat-1]

            for col in ntl_columns:
                if annual:
                    temporal_id = temporal_dummy_ids["_".join(col.split("_")[-1:])] + len(all_params_df) + 1
                    gdp_col_name = "gdp_"+col.split("_")[-1]
                    id_ = country_code + "_" + col.split("_")[-1]
                else:
                    temporal_id = temporal_dummy_ids["_".join(col.split("_")[-2:])] + len(all_params_df) + 1
                    gdp_col_name = "gdp_"+"_".join(col.split("_")[-2:])
                    id_ = country_code + "_" + "_".join(col.split("_")[-2:]) 

                coef_temporal_dummy = reg_model.coef_[0][temporal_id]
                log_ntl, log_gdp = np.log(country_info[col].values[0]), np.log(country_info[gdp_col_name].values[0])
                log_gdp_hat = beta*log_ntl + coef_country_dummy + coef_temporal_dummy #NTL based log GDP

                
                df_for_lambda.append([id_, log_gdp, log_ntl, coef_country_dummy+coef_temporal_dummy, log_gdp_hat, spi_error_std, cent_error_std])

        df_for_lambda = pd.DataFrame(df_for_lambda, columns=["Region_temporal", "log_gdp", "log_ntl", "dummy_vars_sum", "log_gdp_hat", "spi_error_std", "cent_error_std"])

        return df_for_lambda, beta

    @staticmethod
    def evaluate_lambda_for_one_row(row, beta, theta_0, theta_1, theta_2):
        y, spi_error_std, cent_error_std = row[["log_gdp", "spi_error_std", "cent_error_std"]]
        z, a = row[["log_ntl", "dummy_vars_sum"]]#z is log_ntl, a is the sum of dummy variables' coeffs sum.
        e_y_star, e_y_star_2, e_y_star_3 = [y, y**2 + spi_error_std**2, y**3 + 3*y*(spi_error_std**2)] #Expectation of y*, y*^2, y*^3

        lambda_num = spi_error_std**2 #lambda numerator
        lambda_den = spi_error_std**2 #lambda denominator
        lambda_den += (beta*z)**2 + a**2 + 2*beta*z*a #adding E[y_hat^2]
        lambda_den += y**2 + spi_error_std**2 #adding E[y_*^2]
        tmp = (beta*z + a)*e_y_star #expectation of y_hat * y*

        ##Not using the definition in the next line since y_hat is also a given value so E[y_hat . y*] should be y_hat*E[y*].
        # tmp = beta*(theta_0*e_y_star + theta_1*e_y_star_2 + theta_2*e_y_star_3) + a*e_y_star #expectation of y_hat * y*

        lambda_den -= 2*tmp #adding -2E[y^.y*]
        return round(lambda_num/lambda_den, 3)


    def evaluate_lambda(self):
        theta_0, theta_1, theta_2 = self.mle_params["x"][0:3]
        ##Preparing annual data
        df_for_lambda_annual, beta_annual = self.prepare_data_for_lambda_evaluation()

        ##Evaluating lambda using the equation 8 in the IMF paper.
        df_for_lambda_annual["lambda"] = df_for_lambda_annual.apply(self.evaluate_lambda_for_one_row, axis=1, args=(beta_annual, theta_0, theta_1, theta_2))

        ##Preparing monthly data
        df_for_lambda_monthly, beta_monthly = self.prepare_data_for_lambda_evaluation(annual=False)

        ##Evaluating monthly lambda using interpolation from the annual lambda data.
        def tmp_func(row, annual_df):
            region_temporal = row["Region_temporal"]
            if int(region_temporal.split("_")[-1]) == 12:
                return annual_df[annual_df["Region_temporal"] == "_".join(region_temporal.split("_")[0:-1])]["lambda"].values[0]
            else:
                return np.NaN

        ##Assigning year end month the lambda value obtained in df_for_lambda_annual
        df_for_lambda_monthly["lambda"] = df_for_lambda_monthly.apply(tmp_func, axis=1, args=(df_for_lambda_annual, ))

        ##Interpolating annual lambda to get monthly lambda
        df_for_lambda_monthly["Region Code"] = df_for_lambda_monthly.apply(lambda x:x["Region_temporal"].split("_")[0], axis=1)

        def group_interpolation(group):
            group["lambda"] = group["lambda"].interpolate(method="linear", axis=0, limit_direction="both")
            return group

        df_for_lambda_monthly = df_for_lambda_monthly.groupby("Region Code").apply(group_interpolation)
        df_for_lambda_monthly.drop(columns="Region Code", inplace=True)
        return df_for_lambda_monthly

    def evaluate_gdp_monthly(self):
        df = self.evaluate_lambda()
        df["log_gdp_revised"] = df["lambda"]*df["log_gdp_hat"] + (1.0 - df["lambda"])*df["log_gdp"]
        df.to_csv("./data/revised_gdp_{}_monthly.csv".format(self.data_type), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Arguments for linear combination code.')
    parser.add_argument("--dmsp_ols", required=True, type=int, help="1 if monthly GDP is to be evaluated for DMSP-OLS(1992-2012), 0 if on VIIRS data(2013-2021)")

    args = parser.parse_args()
    args.dmsp_ols = bool(args.dmsp_ols)

    linear_combination = LinearCombination(dmsp_ols=args.dmsp_ols)