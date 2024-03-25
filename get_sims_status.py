from glob import glob
import numpy as np
import argparse

def get_missing_sim_files(all_sim_files):
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
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("--seed", required=True, nargs = '+', type=int, help="random seed for which status should be checked.")
    args = parser.parse_args()

    all_min_radius = [10.0, 25.0]
    all_min_radius.extend(np.arange(50.0, 350.0, 50.0))
    all_seeds = args.seed
    all_sim_files = get_required_sim_files(all_min_radius, all_seeds)
    missing_data = get_missing_sim_files(all_sim_files)

    for key, data in missing_data.items():
        count = sum([len(month_data) for month_data in data.values()])
        print(key, count)
