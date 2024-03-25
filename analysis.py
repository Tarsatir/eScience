import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
import os
from tqdm import tqdm
from copy import copy
import rasterio
from contextlib import contextmanager
from rasterio import MemoryFile
from multiprocessing import Process
import argparse

@contextmanager
def get_population_count_tif(year_to_find, month):
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

def get_missing_pop_count(tessellation_data, year, month, radius, seed, key):
    data = tessellation_data[(radius, seed)]
        
    with get_population_count_tif(year, month) as pop_tif:
        pop_data = pop_tif.read()
        pop_data = np.squeeze(pop_data)
        
    rows, cols = [], []

    for pt in  data["site_to_point_dict"][key]:
        rows.append(pt[0])
        cols.append(pt[1])
        
    return np.sum(pop_data[(rows, cols)])


def get_sims_data(radius, seed, tessellation_data):    
    data_dict = {}
    
    missing_pop_count = {}
    
    year_range = np.arange(1992, 2022, 1)
    for year in tqdm(year_range):
        month_range = np.arange(4, 13, 1) if year == 1992 else np.arange(1, 13, 1)
        for month in tqdm(month_range):
            filename = "./data/sims/{}_{}_radius-{}_seed-{}.pkl".format(year, month, radius, seed)
            if not os.path.exists(filename):
                data_dict["{}_{}".format(year, month)] = []
            else:
                with open(filename, "rb") as f:
                    data = pickle.load(f)

                keys = set(list(data["gdp_sum"].keys()))
                tmp = []
                for key in keys:
                    # if key in tessellation_data[(radius, seed)]["site_to_point_dict"]:
                    if data["site_pop_sum"][key] == 1.0:
                        pop_sum = get_missing_pop_count(tessellation_data, year, month, radius, seed, key)
                    else:
                        pop_sum = data["site_pop_sum"][key]
                        
                    tmp.append(round(data["gdp_sum"][key]/pop_sum, 2))

                data_dict["{}_{}".format(year, month)] = tmp
                
        print("year {} done".format(year))

    df = pd.DataFrame.from_dict(data_dict).transpose()
    df.columns = keys
    df.to_csv("./data/results/gdp_{}_{}.csv".format(radius, seed))

def get_combined_tessellation_dict(seed):
    all_tessellation_dict = {}

    for file in glob("./data/*"):
        if "voronoi_tessellation_meta" in file and "{}.pkl".format(seed) in file:
            with open(file, "rb") as f:
                data = pickle.load(f)
            
            radius, seed = file.split("/")[-1].split("_")[-2:]
            radius = float(radius)
            seed = int(seed[0:-4])
            all_tessellation_dict[(radius, seed)] = data

    assert len(all_tessellation_dict) > 0, "no voronoi_tessellation_meta data found."
    
    return all_tessellation_dict



if __name__  == "__main__":
    parser = argparse.ArgumentParser(description='Arguments for simulation code.')
    parser.add_argument("--seed", required=True, type=int, help="random seed to use.")
    parser.add_argument("--n_procs", required=True, type=int, help="num of processes.")
    args = parser.parse_args()

    combined_tessellation_dict = get_combined_tessellation_dict(args.seed)
    print("Combined tessellation dict created.")

    all_radius = [10.0, 25.0, 50.0, 100.0, 150.0, 200.0, 250.0, 300.0]

    n_procs = args.n_procs

    all_processes = []
    for i in range(n_procs):
        process = Process(target=get_sims_data, args=(all_radius[i],  args.seed, combined_tessellation_dict, ))
        all_processes.append(process)

    for process in all_processes:
        process.start()
        
    for process in all_processes:
        process.join()

