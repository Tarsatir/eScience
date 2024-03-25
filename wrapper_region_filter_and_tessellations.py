import numpy as np
from population import *
from voronoi_tessellations import *
from population_based_site_sampling import *
from voronoi_tessellation_data import *
from glob import glob
from contextlib import contextmanager
from tqdm import tqdm
from multiprocessing import Process

@contextmanager
def get_population_count_tif(year_to_find, month):
    # This creates an in-memory population count tif file for a particular year and month. This was used to counter the lack-of-memory problem.
    years = np.arange(1990, 2021, 5)
    
    prev_avail_year, next_avail_year = None, None
    
    if year_to_find == 2021:
        prev_avail_year, next_avail_year = 2015, 2020
    elif year_to_find == 1990:
        prev_avail_year, next_avail_year = 1990, 1995
    else:
        for i in range(len(years)-1):            
            if years[i] <= year_to_find <= years[i+1]:
                prev_avail_year, next_avail_year = years[i], years[i+1]
                break
                
    month_idx = max(0, year_to_find-1-prev_avail_year)*12 + month
                
    first_pop_count_tif = rasterio.open("./data/population/count/{}/population_count.tif".format(prev_avail_year))
    meta_copy = first_pop_count_tif.meta.copy()
    first_pop_count_tif = first_pop_count_tif.read()
    second_pop_count_tif = rasterio.open("./data/population/count/{}/population_count.tif".format(next_avail_year)).read([1])

    first_pop_count_tif = np.squeeze(first_pop_count_tif)
    second_pop_count_tif = np.squeeze(second_pop_count_tif)

    array_m = (second_pop_count_tif - first_pop_count_tif)/60 #represents the slope for linear interpolation
    array_c = second_pop_count_tif - (array_m*60) #represents the intercept for linear interpolation           

    new_pop_count_tif = array_m*np.ones(array_m.shape)*month_idx + array_c
    new_pop_count_tif = np.expand_dims(new_pop_count_tif, axis=0)
    
    with MemoryFile() as memfile:
        with memfile.open(**meta_copy) as dataset:
            dataset.write(new_pop_count_tif)
            del new_pop_count_tif
            
        with memfile.open() as dataset:
            yield dataset

def get_missing_sim_files(all_sim_files):
    # Returns the local filenames of all the simulations file which are missing for all minimum radius and all random seeds.
    collected_data  = {}
    for file_ in glob("./data/sims/*"):
        if ".pkl" not in file_:
            continue
            
        file_ = file_.split("/")[-1]
        year, month, radius, seed = file_.split("_")
        year, month = int(year), int(month)
        radius = float(radius.split("-")[-1])
        seed = int(seed[5:6])
        
        if year not in collected_data:
            collected_data[year] = {month: [str(radius)+"_"+str(seed)]}
        else:
            if month not in collected_data[year]:
                collected_data[year].update({month: [str(radius)+"_"+str(seed)]})
            else:
                collected_data[year][month].append(str(radius)+"_"+str(seed))
                
    missing_data = {}
    for year, data in all_sim_files.items():
        if year not in collected_data:
            missing_data[year] = all_sim_files[year]
            continue
            
        missing_data[year] = {}
        for month, month_data in data.items():
            if month not in collected_data[year]:
                missing_data[year][month] = all_sim_files[year][month]
            else:
                missing_data[year][month] = list(set(all_sim_files[year][month]) - set(collected_data[year][month]))
                
    return missing_data
        

def get_required_sim_files(all_min_radius, all_seeds):
    # Returns the local filenames of all the simulations file which should be there after complete iterations of all minimum radius and all random seeds.
    all_files = {}
    years = np.arange(1992, 2022, 1)
    
    for year in years:
        month_range = np.arange(4, 13, 1) if year == 1992 else np.arange(1, 13, 1)
        year_data = {}
        for month in month_range:
            month_data = []
            for radius in all_min_radius:
                for seed in all_seeds:
                    month_data.append(str(radius)+"_"+str(seed))
            year_data[month] = month_data
        all_files[year] = year_data
    return all_files

def run_sim(years, seeds, missing_data=None):
    print("years selected for the process: ", years)
    print("seeds: ", seeds)
    pop_tif_nodata = -100.0

    cls_population = Population(stitch_images=False, pop_tif_nodata=pop_tif_nodata)
    union_geom = cls_population.get_combined_geometry()

    all_min_radius = [10.0, 25.0]
    all_min_radius.extend(np.arange(50.0, 350.0, 50.0))

    min_radius_to_num_sites_dict = {
        10.0: 200, 25.0: 150, 50.0: 40, 100.0: 30, 150.0: 25, 200.0: 20, 250.0: 15, 300.0: 10
    }

    lon_lat_found = False
    lons, lats = None, None
    country_codes = None
    region_mask = None

    tessellation_for_different_params_dict = {}
    
    for radius in all_min_radius:
        for seed in seeds:
            filename = "./data/voronoi_tessellation_meta_{}_{}.pkl".format(radius, seed) 
            if os.path.exists(filename):
                with open(filename, "rb") as f:
                    tessellation_for_different_params_dict["{}_{}".format(radius, seed)] = pickle.load(f)  

    if missing_data is not None:
        years = sorted(list(missing_data.keys()))

    for year in tqdm(years):
        print("Running sims for year: ", year)
        is_viirs = False if year < 2013 else True

        if missing_data is not None:
            if len(missing_data[year]) == 0:
                continue
            else:
                month_range = []
                for month, month_data in missing_data[year].items():
                    if len(month_data) !=  0:
                        month_range.append(month)

                month_range = sorted(month_range)
        else:
            month_range = np.arange(4, 13 ,1) if year == 1992 else np.arange(1, 13, 1)

        for month in tqdm(month_range):
            month_str = "0"+str(month) if month < 10 else str(month)
            month_int = month

            with get_population_count_tif(year, month_int) as pop_tif:
                for file_ in glob("./data/earth_observation_group/monthly/{}/{}/*".format(year, month_str)):
                    if ".tif" in file_:
                        ntl_tif = rasterio.open(file_)

                filtered_imgs, transforms = cls_population.get_filtered_pop_and_ntl(pop_tif, ntl_tif, union_geom, is_viirs, filter_ntl=False)

                if not lon_lat_found:
                    lons, lats = cls_population.get_geocoordinates(pop_tif.meta, filtered_imgs[1], transforms[1])
                    country_codes = cls_population.get_country_code_for_tif_all(lons, lats)
                    region_mask = ~np.ma.masked_values(country_codes, None).mask
                    lon_lat_found = True

            if missing_data is not None:
                all_min_radius = set([])
                radius_seed_dict = {}
                for val in missing_data[year][month_int]:
                    radius, seed = val.split("_")
                    radius, seed = float(radius), int(seed)
                    all_min_radius.add(radius)
                    if radius not in radius_seed_dict:
                        radius_seed_dict[radius] = [seed]
                    else:
                        radius_seed_dict[radius].append(seed)
                
                all_min_radius = sorted(list(all_min_radius))
            
            for min_radius in tqdm(all_min_radius):
                seed_range = radius_seed_dict[min_radius] if missing_data is not None else seeds

                for seed in seed_range:
                    print("seed: ", seed)
                    num_sites = min_radius_to_num_sites_dict[min_radius]
                    poisson_based=False

                    key = str(min_radius) + "_" +str(seed)
                    # site_to_point_dict =  tessellation_for_different_params_dict[key]["site_to_point_dict"]
                    # sites = tessellation_for_different_params_dict[key]["sites"]


                    if key in tessellation_for_different_params_dict:
                        site_to_point_dict =  tessellation_for_different_params_dict[key]["site_to_point_dict"]
                        sites = tessellation_for_different_params_dict[key]["sites"]
                        print("voronoi tessellation found.")
                    else:
                        voronoi_tessellation = VoronoiTessellation(filtered_imgs[1], region_mask, num_sites, min_radius, [np.min(lats), np.max(lats)], [np.min(lons), np.max(lons)], seed=seed, poisson_based=poisson_based)
                        voronoi_tessellation.run()
                        tessellation_for_different_params_dict[key] = {
                            "site_to_point_dict": voronoi_tessellation.site_to_point_dict, 
                            "sites": voronoi_tessellation.sites}
                        
                        site_to_point_dict = voronoi_tessellation.site_to_point_dict
                        sites = voronoi_tessellation.sites ## sites are stored as (x, y) instead of (y,x)

                    voronoi_data = VoronoiTessellationData(filtered_imgs[1], filtered_imgs[2], country_codes, site_to_point_dict, sites, year, month_int, dmsp_ols=not is_viirs)
                    final_data = voronoi_data.run()

                    with open("./data/sims/{}_{}_radius-{}_seed-{}.pkl".format(year, month_int, min_radius, seed), "wb") as f:
                        pickle.dump(final_data, f)

                    filename = "./data/voronoi_tessellation_meta_{}_{}.pkl".format(min_radius, seed)
                    if not os.path.exists(filename):
                        with open(filename, "wb") as f:
                            pickle.dump(tessellation_for_different_params_dict[key], f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Arguments for simulation code.')
    # parser.add_argument("--stitch_images", required=False, type=int, help="1 if population count images should be stitched.")
    parser.add_argument("--start_year", required=False, type=int, help="start year for simulations.")
    parser.add_argument("--end_year", required=False, type=int, help="end year for simulations.")
    parser.add_argument("--n_procs", required=False, type=int, help="num processes to spawn.")
    parser.add_argument("--resume", required=False, type=int, help="1 if sims should resume for the remainder years.")
    parser.add_argument("--seed", required=True, nargs='+', type=int, help="random seed(s) to use.")

    args = parser.parse_args()

    years = np.arange(1992, 2022, 1) if (args.start_year is None or args.end_year is None) else np.arange(args.start_year, args.end_year+1, 1)
    n_procs = 8 if args.n_procs is None else args.n_procs

    resume = 0 if args.resume is None else args.resume

    if resume:
        all_min_radius = [10.0, 25.0]
        all_min_radius.extend(np.arange(50.0, 350.0, 50.0))
        all_seeds = args.seed
        print("resuming process for seed: ", all_seeds)
        all_sim_files = get_required_sim_files(all_min_radius, all_seeds)
        missing_data = get_missing_sim_files(all_sim_files)
    else:
        missing_data = None
        all_seeds = args.seed


    all_processes = []
    b = int(np.ceil(len(years)/n_procs))
    counter = 0
    for i in range(n_procs):
        if missing_data is not None:
            process_missing_data = {year: missing_data[year] for year in years[counter: counter+b]}
        else:
            process_missing_data = None

        process = Process(target=run_sim, args=(years[counter: counter+b], all_seeds, process_missing_data, ))
        counter += b
        all_processes.append(process)

    for process in all_processes:
        process.start()
        
    for process in all_processes:
        process.join()





# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description='Arguments for population data manipulation code.')
#     parser.add_argument("--stitch_images", required=False, type=int, help="1 if population count images should be stitched.")

#     args = parser.parse_args()
#     args.stitch_images = bool(args.stitch_images)

#     pop_tif_nodata = -100.0

#     cls_population = Population(stitch_images=args.stitch_images, pop_tif_nodata=pop_tif_nodata)
#     union_geom = cls_population.get_combined_geometry()

#     # ToDo:
#     # 1. Iterate over all months for different spatial scales. 
#     # 2. For every iteration params(temporal and spatial scale), run at least 5 times with different seeds and store GDP data for all seeds.
#     # 3. Modify code so as to not filter out NTL data since it is already cut into the desired shape.
#     # 4. Work on voronoi_tessellation_data/get_total_gdp_for_region.
#     # 5. Write code to combine monthly data for any spatial scale into any temporal scale bigger than a month.
#     # 6. Plot GDP distribution over the complete time period for every spatial scale so as to find the threshold for event.

#     years = np.arange(2013, 2014, 1)
#     all_min_radius = np.arange(50.0, 400.0, 50.0)
#     min_radius_to_num_sites_dict = {
#         50.0: 40, 100.0: 30, 150.0: 25, 200.0: 20, 250.0: 15, 300.0: 10, 350: 5
#     }

#     for year in tqdm(years):
#         lon_lat_found = False
#         lons, lats = None, None
#         print("Running sims for year: ", year)
#         is_viirs = False if year < 2013 else True
#         month_range = np.arange(4, 13 ,1) if year == 1992 else np.arange(1, 13, 1)
        
#         for month in tqdm(month_range):
#             month_str = "0"+str(month) if month < 10 else str(month)
#             month_int = month

#             with get_population_count_tif(year, month_int) as pop_tif:
#                 for file_ in glob("./data/earth_observation_group/monthly/{}/{}/*".format(year, month_str)):
#                     if ".tif" in file_:
#                         ntl_tif = rasterio.open(file_)

#                 filtered_imgs, transforms = cls_population.get_filtered_pop_and_ntl(pop_tif, ntl_tif, union_geom, is_viirs, filter_ntl=False)

#                 if not lon_lat_found:
#                     lons, lats = cls_population.get_geocoordinates(pop_tif.meta, filtered_imgs[1], transforms[1])
#                     lon_lat_found = True

#             print(filtered_imgs[1].shape, filtered_imgs[2].shape)
#             country_codes = cls_population.get_country_code_for_tif_all(lons, lats)

#             region_mask = ~np.ma.masked_values(country_codes, None).mask
#             for min_radius in tqdm(all_min_radius):
#                 for seed in np.arange(1, 2):
#                     num_sites = min_radius_to_num_sites_dict[min_radius]
#                     poisson_based=False

#                     voronoi_tessellation = VoronoiTessellation(filtered_imgs[1], region_mask, num_sites, min_radius, [np.min(lats), np.max(lats)], [np.min(lons), np.max(lons)], seed=seed, poisson_based=poisson_based)
#                     voronoi_tessellation.run()

#                     voronoi_data = VoronoiTessellationData(filtered_imgs[1], filtered_imgs[2], country_codes, voronoi_tessellation.site_to_point_dict, voronoi_tessellation.sites, year, month_int, dmsp_ols=not is_viirs)
#                     final_data = voronoi_data.run()

#                     with open("./data/sims/{}_{}_radius-{}_seed-{}.pkl".format(year, month_int, min_radius, seed), "wb") as f:
#                         pickle.dump(final_data, f)

