import numpy as np
import pandas as pd
#import random
#import PIL
import rasterio
#import shapefile
import geopandas as gpd
import matplotlib.pyplot as plt
#from PIL import Image
#from pyproj import Transformer
from rasterio import features
from rasterio.enums import Resampling
from rasterio.mask import mask, raster_geometry_mask
from rasterio.plot import show
from shapely.geometry import Polygon, MultiPolygon, mapping,Point
#from skimage.measure import block_reduce
#from matplotlib.collections import PatchCollection
#from matplotlib.colors import Normalize
#from matplotlib.colors import LogNorm
#from bridson import poisson_disc_samples
#import matplotlib.patches as patches
#import scienceplots 
import os 
import Tesselation as tess
#import ipykernel
import sys

import tesslib as ts 

def run(av_cid):

    # Read the config file.
    config = ts.load_config('./tesslib/config/config_template.yml')
    # extract sampling configuration
    sampling_config = config["sampling"]
    sample_configs = ts.get_sample_configs(sampling_config,config_file=config["config_file"])

    # Expected attributes per config:
    # * id                  (to define sample file names)
    #   * country codes       (to determine shape to process)
    #   * mode (grid, pdf, none)
    # * repeats
    # * min_distance
    # * max_samples

    #create output directory structures
    out_dir_root = config["output_root_directory"]
    out_dir_name = config["output_directory_name"]
    if out_dir_root is None:
        out_dir_root = "../"

    out_dir_path=ts.set_output_dir(out_dir_root,out_dir_name)

    #Use Population density of selected region to sample points 
    raster_path = '../Data_summary/POP/selection.tif'
    #raster_path = '../Data_summary/GDP/1992/1992GDP.tif'
    probability_gdf = tess.raster_to_spatial_probability_distribution(raster_path, power=2)

    #Create far away points.
    far_points = gpd.GeoDataFrame(geometry=gpd.points_from_xy([180.0,-180,-180,180] ,[90.0, 90,-90,-90]))

    for sconfig in sample_configs:
        if sconfig.id == av_cid:
    #Country code can be found in the folder "Data_summary/country_geometry" in the file "Country_codes.csv"
    #In the same folder we have the shapefiles for the countries as well
    
    # Create a joint shape file out of the country list.
    #country_codes = ["NL1", "BEL", "DEU"]
            combined_geo_df = tess.get_combined_country_geometry(sconfig.country_codes)
            combined_geo_df_boundary = combined_geo_df.geometry[0]

    #tess.visualize_geometry(combined_geo_df) 
    
    #Save the population density of the combined countries
    #tess.get_pop_density_for_geodf(combined_geo_df)
    
            for r in range(sconfig.ensemble_size):
                cid = f'S{sconfig.id}_R{r}'
        
                if sconfig.mode == 'grid':
                    #TODO implement grid sampling
                    points = tess.grid_sampling_with_geodf(size=sconfig.radius)
                elif sconfig.mode == 'density':
                    points = ts.poisson_disk_sampling(probability_gdf, sconfig.mode, weights='probability', radius=sconfig.min_distance, n_samples=sconfig.number_points)
                elif config.mode == 'random':
                    #TODO implement ramdom PD sampling
                    points = ts.poisson_disk_sampling(probability_gdf, sconfig.mode, radius=sconfig.min_distance, filter_geom=combined_geo_df, n_samples=sconfig.number_points)
                else:
                    raise ValueError('specified mode value is not supported')

            #Add far points.
                points = pd.concat([far_points, points], ignore_index=True)

            # Save points to disk.
                points.to_file(f"{out_dir_path}/points_{cid}.geojson", driver='GeoJSON')

        # Visualize.
        #tess.visualize_with_population(points, raster_path)
        #print(len(points))

        # Create Voronoi diagram.
                voronoi_gdf = tess.create_voronoi_gdf(points, combined_geo_df_boundary)
        
        # Save Voronoi to disk.
                voronoi_gdf.to_file(f"{out_dir_path}/voronoi_{cid}.geojson", driver='GeoJSON')

        else:
            pass

if __name__ == "__main__":
    av_cid = sys.argv[1]
    run(av_cid)