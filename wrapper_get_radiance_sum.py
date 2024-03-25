## Wrapper for evaluating annual radiance sum for all years.

from get_radiance_sum import *
import argparse
import glob

# Common code for annual and monthly radiance evaluation
def get_radiance_for_time_period(year, ntl_filepath, region_shapefilepath, north_latitude_limit, south_latitude_limit, n_procs, save=False):
    for rasterio_file in glob.glob(ntl_filepath+"/*"):
        if rasterio_file.split("/")[-1].split(".")[-1] == "tif":
            if year in np.arange(1992, 2012+1):
                satellite_name = rasterio_file.split("/")[-1].split(".")[0][:3]
            
            ntl_filepath = rasterio_file
            ntl_df = run(ntl_filepath, region_shapefilepath, north_latitude_limit, south_latitude_limit, n_procs=n_procs)

            if save:
                if year in np.arange(1992, 2012+1):
                    save_path = "/".join(ntl_filepath.split("/")[0:-1])+"/countries_ntl_sum_"+satellite_name+".csv"
                else:
                    save_path = "/".join(ntl_filepath.split("/")[0:-1])+"/countries_ntl_sum.csv"
                
                ntl_df.to_csv(save_path, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Arguments for NTL yearly aggregation')
    parser.add_argument("--start_year", type=int, help="start year")
    parser.add_argument("--end_year", type=int, help="end year")
    parser.add_argument("--years", nargs="+", type=int, help="years for which annual radiance is to be evaluated")
    parser.add_argument("--n_procs", type=int, help="number of processes to create", required=True)
    parser.add_argument("--save", required=True, type=int, help="1 to save the results")
    parser.add_argument("--monthly", required=False, type=int, help="1 to get radiance sum for monthly data.")
    print(parser.print_help())
    print("_"*100)
    print()
    print("Remove the exit() command in the code after this print statement.")
    exit(0)

    args = parser.parse_args()
    args.save = True if args.save == 1 else False
    args.monthly = True if args.monthly == 1 else False

    if args.start_year is None and args.end_year is None:
        if args.years is None:
            raise Exception("Either provide a list as value to the argument 'years' or provide 'start_year' and 'end_year'.")
        else:
            years = args.years
    else:
        years = [year for year in range(args.start_year, args.end_year+1)]
    
    print("Evaluation will be done for the years: ", years)


    region_shapefilepath = os.getcwd()+"/data/country_geometry/all_country/ne_10m_admin_0_countries.shp"
    
    north_latitude_limit = 75.0
    south_latitude_limit = -65.0

    for year in years:
        print("Evaluating {} radiance sum for year {} ...".format("monthly" if args.monthly else "annual", year))
        ntl_filepath = os.getcwd()+"/data/earth_observation_group/"
        try:
            if not args.monthly:
                ntl_filepath += "annual/"+str(year)+"/"
                get_radiance_for_time_period(year, ntl_filepath, region_shapefilepath, north_latitude_limit, south_latitude_limit, n_procs=args.n_procs, save=args.save)
            else:
                ntl_filepath += "monthly/"+str(year)+"/"
                for month_folder in glob.glob(ntl_filepath+"*"):
                    get_radiance_for_time_period(year, month_folder, region_shapefilepath, north_latitude_limit, south_latitude_limit, n_procs=args.n_procs, save=args.save)
                
                    print("Evaluation for month {} for year {} done".format(month_folder.split("/")[-1], year))

            print("Evaluation for year {} done.".format(year))
            print("_"*10)
        except Exception as e:
            print("Exception: {}, caught for year: {}".format(e, year))
            continue

