import rasterio
from rasterio.mask import mask
import geopandas as gpd
from shapely.geometry import mapping
import numpy as np
import pandas as pd
from tqdm import tqdm
from multiprocessing import Process, Manager, Value, Pool
import copy
import pickle
import argparse
import os
import csv


def check_polar_limit(geom_coords, north_latitude_limit, south_latitude_limit):
    for coord in geom_coords:
        if coord[1] > north_latitude_limit or coord[1] < south_latitude_limit:
            return True
    return False

## Gets polar regions where NTL satellites don't have data.
def get_polar_regions(north_latitude_limit, south_latitude_limit, country_geometry_dict):
    polar_regions = []
    
    for country, country_geometry in country_geometry_dict.items():
        if country_geometry.geom_type == "MultiPolygon":
            for i in range(len(country_geometry.geoms)):
                geom_coords = country_geometry.geoms[i].exterior.coords[::-1]
                if check_polar_limit(geom_coords, north_latitude_limit, south_latitude_limit):
                    polar_regions.append(country)
                    break
                
        else:
            geom_coords = country_geometry.exterior.coords[::-1]
            if check_polar_limit(geom_coords, north_latitude_limit, south_latitude_limit):
                polar_regions.append(country)
                break
                
    return polar_regions

## Getting masked polygons(preprocessing for radiance sum evaluation)
def get_masked_polygons(rasterio_img, country_geometry_dict):
    country_mappings = {country: mapping(country_geometry) for country, country_geometry in country_geometry_dict.items()}
    country_masked_polygons = dict()
    for country, country_geometry in tqdm(country_mappings.items()):
        out_image, _ = mask(rasterio_img, [country_geometry], crop=True)
        country_masked_polygons[country] = out_image
    
    return country_masked_polygons


## Getting radiance value inside a polygon defined by geocoordinates
def get_polygon_radiance_value(masked_region_image, polygon, no_rasterio_data):
    try:
#         # transform to GeJSON format
#         geoms = [mapping(polygon)]

#         # extract the raster values values within the polygon
#         out_image, out_transform = mask(rasterio_img, geoms, crop=True)

#         # no data values of the original raster
#         no_data=rasterio_img.nodata

#         del rasterio_img

        # extract the values of the masked array
        data = np.squeeze(masked_region_image)
#         data = np.reshape(masked_region_image.data.tolist(), (masked_region_image.shape[1], masked_region_image.shape[2]))

        if no_rasterio_data is not None:
            # print("rasterio data has points where data is missing. Check 'no_rasterio_data'")
            # extract the row, columns of the valid values
            row, col = np.where(data != no_rasterio_data) 
            elev = np.extract(data != no_rasterio_data, data)

            num_pixels = elev.shape[0]*elev.shape[1]
            rad_sum = round(np.sum(elev), 2)
            rad_sum_sq = round(np.sum(np.square(elev)), 2)
        else:
            num_pixels = data.shape[0]*data.shape[1]
            rad_sum = round(np.sum(data), 2)
            rad_sum_sq = round(np.sum(np.square(data)), 2)
    except Exception as error:
        print("Error: {} while evaluating radiance sum.".format(error))
        rad_sum = None
        rad_sum_sq = None
        num_pixels = None
    return [rad_sum, rad_sum_sq, num_pixels]

def get_polygon_radiance_value_for_mproc(region_polygon_dict, region_masked_polygon_dict, no_rasterio_data, ret_dict, process_number):
    for region, polygon in (pbar:=tqdm(region_polygon_dict.items())):
        pbar.set_description(f"Process: {process_number}")
        try:
            print("region: ", region)
            res = get_polygon_radiance_value(region_masked_polygon_dict[region], polygon, no_rasterio_data)
            ret_dict[region] = res
        except Exception as error:
            print("Error with Process: {}, e: {}".format(process_number, error))
            ret_dict[region] = [None, None, None]
        
#     del rasterio_img    
#     print("Process: {} complete.".format(process_number))


def get_region_geometry_dict(filepath, north_latitude_limit, south_latitude_limit):
    all_region_shapefile = gpd.read_file(filepath)

    # extract the geometry in GeoJSON format
    all_region_geometry = all_region_shapefile.geometry.values # list of shapely geometries
    region_names = list(all_region_shapefile.SOV_A3)
    region_geometry_dict = {region_names[i]: all_region_geometry[i] for i in range(len(all_region_geometry))}
    
    for key in get_polar_regions(north_latitude_limit, south_latitude_limit, region_geometry_dict):
        del region_geometry_dict[key]
        
    ## Region 'KIR' is creating an issue
    if "KIR" in region_geometry_dict:
        del region_geometry_dict["KIR"]
        
    return region_geometry_dict

def run(ntl_filepath, region_shapefilepath, north_latitude_limit, south_latitude_limit, n_procs=7):
    print("Getting country geometries ...")
    region_geometry_dict = get_region_geometry_dict(region_shapefilepath, north_latitude_limit, south_latitude_limit)
    
    rasterio_img = rasterio.open(ntl_filepath)
    print("Getting country mappings ...")
    region_masked_polygons = get_masked_polygons(rasterio_img, region_geometry_dict)
    
    assert len(region_geometry_dict) == len(region_masked_polygons), "Masked polygons could not be created for all the regions."
    
    # no data values of the original raster
    no_rasterio_data=rasterio_img.nodata
    
    del rasterio_img
    
#     with open("regions_selected.csv", "w") as f:
#         csv_writer = csv.writer(f)

#         for val in list(region_geometry_dict.keys()):
#             csv_writer.writerow([val])
    

    regions_per_proc = len(region_geometry_dict)//n_procs
    all_regions = list(region_geometry_dict.keys())
    extra_keys = len(all_regions) - (n_procs * regions_per_proc)
    prev_start = 0

    manager = Manager()
    all_region_rad_sum_dict = manager.dict()

    all_processes = []
    
    all_arguments = []

    for i in range(n_procs):
        if i == 0:
            regions_for_process = all_regions[prev_start: prev_start + regions_per_proc+extra_keys]
            prev_start = prev_start + regions_per_proc+extra_keys
        else:
            regions_for_process = all_regions[prev_start: prev_start + regions_per_proc]
            prev_start = prev_start + regions_per_proc


        proc_region_geometry_dict = {region: region_geometry_dict[region] for region in regions_for_process}
        proc_region_masked_polygon_dict = {region: region_masked_polygons[region] for region in regions_for_process}
        
        all_arguments.append([ntl_filepath, proc_region_geometry_dict, i+1])

        process = Process(target=get_polygon_radiance_value_for_mproc, args=(proc_region_geometry_dict, proc_region_masked_polygon_dict, no_rasterio_data, all_region_rad_sum_dict, i+1))
        all_processes.append(process)
        
#     with Pool(n_procs) as pool:
#         for result in pool.starmap(get_polygon_radiance_value_for_mproc, all_arguments):
#             print(len(result))
    
    for process in all_processes:
        process.start()
    
    for process in all_processes:
        process.join()
    
    print("total regions: ", len(all_region_rad_sum_dict))
    
#     with open("regions_evaluated.csv", "w") as f:
#         csv_writer = csv.writer(f)

#         for val in list(all_region_rad_sum_dict.keys()):
#             csv_writer.writerow([val])

    ntl_df = pd.DataFrame()
    ntl_df["Region Code"] = all_region_rad_sum_dict.keys()
    ntl_values = np.array(list(all_region_rad_sum_dict.values()))
    ntl_df["annual_ntl_sum"] = ntl_values[:, 0]
    ntl_df["annual_ntl_sum_sq"] = ntl_values[:, 1]

    country_pixels_file = "./data/country_num_pixels.pkl"
    if not os.path.exists(country_pixels_file):
        with open(country_pixels_file, "wb") as f:
            tmp_dict = {key: value[2] for key, value in all_region_rad_sum_dict.items()}
            pickle.dump(tmp_dict, f)

    return ntl_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Arguments for NTL yearly aggregation')
    parser.add_argument('--year', required=True, type=int, help='year for which NTL yearly aggregation should be done.')
    parser.add_argument('--month', required=True, type=str, help='if aggregation is monthly then month number')
    parser.add_argument('--ntl_filename', required=True, type=str, help='filename of the NTL data')
    parser.add_argument('--n_procs', required=True, type=int, help='number of processes')
    parser.add_argument("--save", required=True, type=int, help="save the results. 1 is true, 0 is false.")
    args = parser.parse_args()

    parser.print_help()
    print("_"*100)
    print()
    print("Remove the exit() command in the code after this print statement.")
    exit(0)

    if args.month == "":
        args.month = None

    ntl_filepath = os.getcwd()+"/data/earth_observation_group/"

    if args.month is not None:
        ntl_filepath += "monthly/"+str(args.year)+"/"+str(args.month)+"/"
    else:
        ntl_filepath += "annual/"+str(args.year)+"/"
        
        
    ntl_filepath += args.ntl_filename

    region_shapefilepath = os.getcwd()+"/data/country_geometry/all_country/ne_10m_admin_0_countries.shp"
    
    north_latitude_limit = 75.0
    south_latitude_limit = -65.0
    
    ntl_df = run(ntl_filepath, region_shapefilepath, north_latitude_limit, south_latitude_limit, n_procs=args.n_procs)

    if args.save == 1:
        ntl_df.to_csv(os.getcwd()+"/data/earth_observation_group/annual/{}/countries_ntl_sum.csv".format(args.year), index=False)
