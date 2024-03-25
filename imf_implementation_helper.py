import os
import pandas as pd
import numpy as np
import glob
import copy
from sklearn.linear_model import LinearRegression


# returns the num of days in the month
def get_multiplication_factor(month, year):
    if month == 2:
        if year % 4 == 0:
            return 29.0
        else:
            return 28.0
    else:
        return 31.0 if month in [1, 3, 5, 7, 8, 10, 12] else 30.0
    

## Combines all years NTL data and evaluates NTL per capita
def get_all_ntl_countries_data(years, pop_df, annual=True):
    combined_ntl_df = pd.DataFrame()
    
    for year in years:
        if annual:
            base_path = os.getcwd()+"/data/earth_observation_group/annual/{}".format(year)
            csv_files = [file for file in glob.glob(base_path+"/*") if ".csv" in file]
            if len(csv_files) > 1:
                csv_files = [base_path+"/countries_ntl_sum.csv"]

        else:
            base_path = os.getcwd()+"/data/earth_observation_group/monthly/{}".format(year)
            csv_files = []
            for month_folder in glob.glob(base_path+"/*"):
                tmp = [file for file in glob.glob(month_folder+"/*") if ".csv" in file]
                if len(tmp) > 1:
                    csv_files.append(month_folder+"/countries_ntl_sum.csv")
                else:
                    csv_files.append(tmp[0])
                
        multiplication_factor = 365.0
        ntl_column_name = "rescaled_calibrated_ntl_sum" if (not annual and year < 2013) else "annual_ntl_sum"
        for file in csv_files:
            if not annual:
                month = file.split("/")[-2]
                month = int(month[1]) if month[0] == '0' else int(month)
                multiplication_factor = get_multiplication_factor(month, year)

            ntl_df = pd.read_csv(file)
            ntl_df = ntl_df[["Region Code", ntl_column_name]]

            ntl_df[ntl_column_name] = ntl_df[ntl_column_name]*multiplication_factor
            
            ntl_df = ntl_df.set_index("Region Code").join(pop_df.set_index("Country Code")[str(year)])
            ntl_df[ntl_column_name] = ntl_df[ntl_column_name]/ntl_df[str(year)]
            ntl_df[ntl_column_name] = ntl_df[ntl_column_name].where(ntl_df[ntl_column_name]>0.0, other=np.NaN)
            # ntl_df = ntl_df[ntl_df[ntl_column_name] >= 0.0]
            ntl_df.drop(columns=str(year), inplace=True)

            if not annual:
                if year  < 2013:
                    ntl_df.rename(columns={ntl_column_name: "{}_per_capita_{}_{}".format(ntl_column_name, year, month)}, inplace=True)
                else:
                    ntl_df.rename(columns={"annual_ntl_sum": "{}_per_capita_{}_{}".format("rescaled_calibrated_ntl_sum", year, month)}, inplace=True)
            else:
                ntl_df.rename(columns={ntl_column_name: "{}_per_capita_{}".format(ntl_column_name, year)}, inplace=True)
            
            ntl_df.reset_index(inplace=True)

            if combined_ntl_df.empty:
                combined_ntl_df = copy.copy(ntl_df)   
            else:
                combined_ntl_df = combined_ntl_df.set_index("Region Code").join(ntl_df.set_index("Region Code"))
                combined_ntl_df.reset_index(inplace=True)

    ## Dropping the rows which have more than 50% missing values
    combined_ntl_df = combined_ntl_df[~(combined_ntl_df.isna().mean(axis=1)*100.0 > 50.0)]

    if not annual:
        cols = set(combined_ntl_df.columns)
        ## Adding missing month columns(NTL was not found for these months)
        for year in years:
            month_range = range(4, 12+1) if year == 1992 else range(1, 12+1)
            for month in month_range:
                col_name = "rescaled_calibrated_ntl_sum_per_capita_{}_{}".format(year, month)
                if col_name not in cols:
                    combined_ntl_df[col_name] = [None]*len(combined_ntl_df)
                    combined_ntl_df[col_name] = combined_ntl_df[col_name].astype(np.dtype(np.float64), copy=True)


        ## Sorting the columns in the increasing order of year and month.
        cols = list(combined_ntl_df.columns)
        sorted_cols = sorted(cols[1:], key=lambda x: (int(x.split("_")[-2]), int(x.split("_")[-1])))
        sorted_cols.insert(0, cols[0])
        combined_ntl_df = combined_ntl_df[sorted_cols]

        ## Imputing missing NTL values using linear interpolation.
        used_cols = list(combined_ntl_df.columns)[1:]
        combined_ntl_df[used_cols] = combined_ntl_df[used_cols].interpolate(method="linear", axis=1, limit_direction="both")


    combined_ntl_df.sort_values(by=["Region Code"], inplace=True)
    combined_ntl_df.reset_index(inplace=True, drop=True)
    return combined_ntl_df


def get_spi_and_centroid_df(spi_centroid_filename):
    spi_and_centroid_df = pd.read_csv(spi_centroid_filename)
    columns_to_choose = ["iso3c", "spi_category", "centroid_category"]
    spi_and_centroid_df = spi_and_centroid_df[columns_to_choose]
    spi_and_centroid_df.rename(columns={"iso3c": "Region Code"}, inplace=True)
    return spi_and_centroid_df

def get_gdp_df(gdp_filename, years):
    gdp_df = pd.read_csv(gdp_filename)

    columns_to_choose = ["Country Code"]
    cols = gdp_df.columns[2:]
    years_set = set(years)
    for col in cols:
        if int(col[0:4]) in years_set:
            columns_to_choose.append(col)

    gdp_df = gdp_df[columns_to_choose]

    new_column_names = {col: "gdp_"+col for col in list(gdp_df.columns)[1:]}
    new_column_names["Country Code"] = "Region Code"
    gdp_df.rename(columns=new_column_names, inplace=True)
    return gdp_df

def merge_all_data(combined_ntl_df, spi_and_centroid_df, gdp_df, is_viirs_data, annual=True):
    all_params_df = pd.merge(gdp_df, combined_ntl_df, on="Region Code")
    all_params_df = all_params_df.merge(spi_and_centroid_df, on="Region Code")
    all_params_df = all_params_df.dropna().reset_index(drop=True)

    save_filename = os.getcwd()+"/data/imf_data_combined_{}{}.csv".format("viirs" if is_viirs_data else "dmsp_ols", "" if annual else "_monthly")
    all_params_df.to_csv(save_filename, index=False)
    return all_params_df

def create_one_hot_ntl_data(df, is_viirs_data=True, annual=True):
    df.sort_values(by=["Region Code"], inplace=True)
    gdp_columns = [col for col in df.columns if "gdp" in col]
    gdp_columns
    if annual:
        gdp_columns = sorted(gdp_columns, key=lambda x: int(x.split("_")[-1]))
    else:
        gdp_columns = sorted(gdp_columns, key=lambda x: (int(x.split("_")[-2]), int(x.split("_")[-1])))
    
    num_spatial_dummies = len(df)
    num_temporal_dummies = len(gdp_columns)
            
    X = []
    y = []
    
    i = 0
    for col in gdp_columns:
        x_col = None
        if annual:
            x_col = "annual_ntl_sum_per_capita_"+col.split("_")[1]
        else:
            x_col = "rescaled_calibrated_ntl_sum_per_capita_"+"_".join(col.split("_")[1:])
        
        if x_col not in df.columns:
            continue
        ntl_values = np.array(np.log(list(df[x_col]))).reshape((len(df), 1))
        spatial_dummies = np.identity(num_spatial_dummies, dtype=np.float64)
        temporal_dummies = np.zeros((len(df), num_temporal_dummies), dtype=np.float64)
        temporal_dummies[:, i] = [1.0]*len(df)
        gdp_values = np.array(np.log(list(df[col]))).reshape((len(df), 1))
        
        
        new_x_values = np.hstack((ntl_values, spatial_dummies, temporal_dummies))
        X = new_x_values if len(X) ==  0 else np.vstack((X, new_x_values))
        y = gdp_values if len(y) == 0 else np.vstack((y, gdp_values))
                
        i+=1
    
    return X, y


def get_ntl_gdp_model(X, y):
    model = LinearRegression(fit_intercept=False).fit(X, y)
    return model


if __name__ == "__main__":
    pop_df = pd.read_csv(os.getcwd()+"/data/population/API_SP.POP.TOTL_DS2_en_csv_v2_5358404.csv")
    years = np.arange(2013, 2021+1)
    combined_ntl_df = get_all_ntl_countries_data(years, pop_df, annual=False)
    
    ## Reading and processing SPI and Centroid data.
    spi_centroid_filename = os.getcwd()+"/data/SPI_index_mean_and_centroid_latitude.csv"
    spi_and_centroid_df = get_spi_and_centroid_df(spi_centroid_filename)

    ## Reading and processing GDP per capita data.
    # gdp_filename = os.getcwd()+"/data/gdp/gdp_per_capita.csv"
    gdp_filename = os.getcwd()+"/data/gdp/monthly_gdp_per_capita.csv"
    gdp_df = get_gdp_df(gdp_filename, years)

    ## Combining all data
    all_params_df = merge_all_data(combined_ntl_df, spi_and_centroid_df, gdp_df, is_viirs_data=True, annual=False)