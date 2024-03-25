## Implementation of gathering data from Voronoi tessellations
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm

class VoronoiTessellationData:
    def __init__(self, pop_data, ntl_data, country_codes_data, site_to_point_dict, site_location_dict, year, month, dmsp_ols=True):
        self.pop_data = pop_data
        self.ntl_data = ntl_data #This data is not per capita. It represents the complete brightness for a cell.

        ## Setting values less than 0 to 0.
        self.ntl_data[np.where(self.ntl_data < 0)] = 0

        self.country_codes = list(np.unique(country_codes_data[np.where(country_codes_data != None)]))
        self.country_codes_data = country_codes_data
        self.site_to_point_dict = site_to_point_dict
        self.site_location_dict = site_location_dict
        self.year = year
        self.month = month
        self.dmsp_ols = dmsp_ols


    def load_ntl_gdp_params(self):
        with open("./data/ntl_gdp_models.pkl", "rb") as f:
            ntl_gdp_models = pickle.load(f)

        ntl_gdp_model = ntl_gdp_models["{}_monthly".format("dmsp_ols" if self.dmsp_ols else "viirs")]
        return ntl_gdp_model

    def load_gdp_data(self):
        filename = "./data/imf_data_combined_{}_monthly.csv".format("dmsp_ols" if self.dmsp_ols else "viirs")
        return pd.read_csv(filename).sort_values(by=["Region Code"])

    def load_official_gdp_data(self):
        return pd.read_csv("./data/gdp/monthly_gdp_per_capita.csv")

    def load_lambda_data(self):
        return pd.read_csv("./data/revised_gdp_{}_monthly.csv".format("dmsp_ols" if self.dmsp_ols else "viirs"))

    def get_reg_params_for_countries(self):
        ntl_columns = [col for col in self.imf_data.columns if "ntl" in col]
        ntl_columns = sorted(ntl_columns, key=lambda x: (int(x.split("_")[-2]), int(x.split("_")[-1])))
        temporal_dummy_ids = {"_".join(col.split("_")[-2:]): i for i, col in enumerate(ntl_columns)}
        temporal_id = temporal_dummy_ids["{}_{}".format(self.year, self.month)] + len(self.imf_data) + 1
        temporal_dummy = self.ntl_gdp_model.coef_[0][temporal_id]

        reg_params = {}

        for country_code in self.country_codes:
            country_info =  self.imf_data[self.imf_data["Region Code"] == country_code]
            country_id = country_info.index[0] + 1 # Added 1 because the first value in regression params is beta.
            coef_country_dummy = self.ntl_gdp_model.coef_[0][country_id]

            reg_params[country_code] = coef_country_dummy
        
        reg_params["temporal"] = temporal_dummy
        reg_params["beta"] = self.ntl_gdp_model.coef_[0][0]

        return reg_params

    def get_multiplication_factor(self):
        if self.month == 2:
            if self.year % 4 == 0:
                return 29.0
            else:
                return 28.0
        else:
            return 31.0 if self.month in [1, 3, 5, 7, 8, 10, 12] else 30.0


    def get_total_gdp_for_region(self, site_pts):
        
        pts = (site_pts[:, 0], site_pts[:, 1])
        region_pop_data = self.pop_data[pts]
        region_pop_data[np.where(region_pop_data <= 0.0)] = 1.0 #doing this is to avoid division by zero error.

        country_codes_data = self.country_codes_data[pts]

        lambda_keys = np.char.add(country_codes_data.astype(str), "_"+str(self.year)+"_"+str(self.month))
        lambda_ = []
        spatial_reg_params = []
        log_official_gdp_per_capita = []

        lambda_dict = {val[0]: val[1] for _, val in self.lambda_data[["Region_temporal", "lambda"]].iterrows()}
        lambda_ = list(map(lambda_dict.get, list(lambda_keys)))
        spatial_reg_params = list(map(self.reg_params.get, list(country_codes_data)))

        offical_gdp_dict = {val[0]: np.log(val[1]) for _, val in self.offical_gdp_data[["Country Code", "{}_{}".format(self.year, self.month)]].iterrows()}

        log_official_gdp_per_capita = list(map(offical_gdp_dict.get, list(country_codes_data)))

        lambda_ = np.array(lambda_)
        lambda_ = np.squeeze(lambda_)
        spatial_reg_params = np.array(spatial_reg_params)

        ntl_per_capita = self.ntl_data[pts] * self.get_multiplication_factor()/region_pop_data
        ntl_per_capita = np.array(ntl_per_capita)
        ntl_per_capita[np.where(ntl_per_capita <= 0.0)] = 1.0 #doing this is to avoid division by zero error.
        
        log_ntl_per_capita = np.log(ntl_per_capita) #log NTL per capita for whole month
        log_ntl_based_gdp_per_capita = self.reg_params["beta"]*log_ntl_per_capita + spatial_reg_params + self.reg_params["temporal"] #log NTL based GDP per capita for whole month
        
        log_official_gdp_per_capita = np.array(log_official_gdp_per_capita) #log Official GDP per capita for whole month
        log_official_gdp_per_capita = np.squeeze(log_official_gdp_per_capita)

        del country_codes_data, lambda_keys, spatial_reg_params, log_ntl_per_capita, ntl_per_capita


        log_real_gdp_per_capita = lambda_*log_ntl_based_gdp_per_capita + (1-lambda_)*log_official_gdp_per_capita

        real_gdp_per_capita = np.exp(log_real_gdp_per_capita)
        real_gdp = real_gdp_per_capita * region_pop_data

        del lambda_, log_ntl_based_gdp_per_capita, log_official_gdp_per_capita, region_pop_data
        return real_gdp

    def get_total_gdp(self):
        site_real_gdp_sum = {}
        site_pop_sum = {}
        all_site_pts = []
        for site, site_pts in self.site_to_point_dict.items():
            all_site_pts.extend(site_pts)
        
        all_site_pts = np.array(all_site_pts)
        real_gdp = self.get_total_gdp_for_region(all_site_pts)

        counter = 0
        for site, site_pts in self.site_to_point_dict.items():
            site_real_gdp_sum[site] = np.sum(real_gdp[counter:counter+len(site_pts)])
            counter += len(site_pts)
            site_pts = np.array(list(site_pts))
            site_pop_sum[site] = np.sum(self.pop_data[(site_pts[:, 0], site_pts[:, 1])])

            if site_pop_sum[site] == 0.0:
                site_pop_sum[site] = 1.0

        return site_real_gdp_sum, site_pop_sum

    def run(self):
        # loading data
        self.ntl_gdp_model = self.load_ntl_gdp_params()
        self.imf_data = self.load_gdp_data()

        self.offical_gdp_data = self.load_official_gdp_data()
        self.lambda_data = self.load_lambda_data()
        self.reg_params = self.get_reg_params_for_countries()

        # Getting region wise total real GDP
        site_real_gdp_sum, site_pop_sum = self.get_total_gdp()
        data_to_save = {
            "site_location": self.site_location_dict,
            "gdp_sum": site_real_gdp_sum,
            "site_pop_sum": site_pop_sum
        }
        
        return data_to_save
