# Downloading NTL data
NTL data is available in two different formats:
1. From 1992-2012, DMSP-OLS project was collecting it and the values are available in Digital Number(0-255) [DMSP-OLS data](https://eogdata.mines.edu/products/dmsp/)
2. From 2013 onwards, VIIRS project was collecting it and the values are measure in radiance of light [VIIRS data](https://eogdata.mines.edu/products/vnl/#v1)

***To download yearly data***: Currently this process of downloading is not automated. Data can be downloaded manually by visiting the hyperlinks provided above.

***To download monthly data***: Use download_monthly_ntl.py

# DMSP-OLS NTL data post-processing
## Combining different satellite data for DMSP-OLS type of NTL data
For the same time period(be it year or month), NTL data from DMSP-OLS project might have multiple satellites data. For our project, we take the mean value of NTL and combine it into one data file. Use dmsp-ols_data_processing.py for this

## Intercalibration of data
Read about the need for intercalibration [here](https://eogdata.mines.edu/products/dmsp/). To perform it use dmsp-ols_data_processing.py

# Getting NTL sum for a region
1. If region is a country: Use wrapper_get_radiance_sum.py.
2. If region is sub-national or supra-national, the above code can be used but minor modifications are required. Need to have region polygon geometry for that.

# Downloading GDP, Statistical Performance Index(SPI), country-wise population and high-resolution(like population count for a city or even smaller) data
 1. ***GDP***: ./data/API_NY.GDP.MKTP.CD_DS2_en_csv_v2_4901850.csv . It can be found [here](https://databank.worldbank.org/reports.aspx?source=2&series=NY.GDP.MKTP.CD&country=#) as well.
 2. ***SPI***: ./data/SPI_index.csv . It can be found [here](https://www.worldbank.org/en/programs/statistical-performance-indicators) as well.
 3. ***Country-wise population***: ./data/API_SP.POP.TOTL_DS2_en_csv_v2_5358404.csv . It can be found [here](https://databank.worldbank.org/reports.aspx?source=2&series=SP.POP.TOTL&country=)as well.
 4. ***High-resoltion population***: ./data/population/count/ . It can be found [here](https://ghsl.jrc.ec.europa.eu/download.php?ds=pop) as well.


 # Getting GDP based on official reported and NTL based GDP(new GDP measure) as discussed [here](https://www.imf.org/en/Publications/WP/Issues/2019/04/09/Illuminating-Economic-Growth-46670): 
 Use imf_implementation_linear_combination.py for this. This file gives monthly GDP based on officially reported annual GDP and NTL-based GDP.

 # Creating voronoi tessellations and evaluating new GDP measure for these tessellations:
 Use wrapper_region_filter_and_tessellations.py


 For more clarification you can contact me at i.a.mengesha@uva.nl