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

    raster_path = '../Data_summary/POP/selection.tif'
    out_dir_path = '../Output'
    start_year=1993
    end_year=1997

    # Read the config file.
    config = ts.load_config('./tesslib/config/config_template.yml')
# extract sampling configuration
    sampling_config = config["sampling"]
    sample_configs = ts.get_sample_configs(sampling_config,config_file=config["config_file"])


# Expected attributes per config:
# * id                  (to define sample file names)
# * country codes       (to determine shape to process)
# * mode (grid, pdf, none)
# * repeats
# * min_distance
# * max_samples
    
    # Attribute values that have only one measurement over time
    population_gdf = tess.raster_to_population_gdf(raster_path)

    for sconfig in sample_configs:
        if sconfig.id == av_cid:
            for r in range(sconfig.ensemble_size):
                cid = f'S{config.id}_R{r}'
        
            # Load Voronoi from disk.
                voronoi_gdf=gpd.read_file(f"{out_dir_path}/voronoi_{cid}.geojson")

            # Attribute values.
                attributed_voronoi_gdf = tess.attribute_values_to_voronoi_cells(voronoi_gdf, population_gdf, 'population')

            # Save attributed Voronoi to disk.
                attributed_voronoi_gdf.to_file(f"{out_dir_path}/voronoi_attr_{cid}.geojson", driver='GeoJSON')
        else:
            pass


    years = list(range(start_year, end_year))
    for year in years:
    # Read attribute data for this year.
        raster_path = f'../Data_summary/GDP/{year}/{year}GDP.tif'
        #output_raster_path = f'../Data_summary/POP/gdp_selection_{year}.tif'
        #tess.selection_of_tif_for_geodf(combined_geo_df, raster_path, output_raster_path)
        #gdp_gdf = tess.raster_to_gdf(output_raster_path, name=f'gdp_{year}')
        gdp_value_name = f'gdp_{year}'  # Dynamic column name for the GDP data
        with rasterio.open(raster_path) as src:

            for sconfig in sample_configs:
                if sconfig.id == av_cid:
                    combined_geo_df = ts.get_region_geometry(config.country_codes)
                    combined_geo_df_boundary = combined_geo_df.geometry[0]
                    if combined_geo_df.crs != src.crs:
                        combined_geo_df = combined_geo_df.to_crs(src.crs)

                    seldata, seldata_transform = rasterio.mask.mask(src,combined_geo_df.geometry,crop=True)

                    for r in range(config.ensemble_size):
                        cid = f'S{config.id}_R{r}'

                        gdp_gdf = ts.gdf_to_raster(seldata,name=gdp_value_name,crs=src.crs)
            
                # Load Voronoi from disk.
                        voronoi_gdf=gpd.read_file(f"{out_dir_path}/voronoi_attr_{cid}.geojson")

                # Attribute values.
                        attributed_voronoi_gdf = tess.attribute_values_to_voronoi_cells(voronoi_gdf, gdp_gdf, gdp_value_name)

                # Save attributed Voronoi to disk.
                        if year == end_year-1 :
                            pars=config.__dict__
                            pars.update({"start_year":start_year,"end_year":year,"ensemble_member":r})
                            attributed_voronoi_gdf=ts.append_metadata_to_gdf(attributed_voronoi_gdf,pars)
                            attributed_voronoi_gdf.to_file(f"{out_dir_path}/voronoi_{cid}_final.geojson", driver='GeoJSON')   
                        else:
                            attributed_voronoi_gdf.to_file(f"{out_dir_path}/voronoi_{cid}.geojson", driver='GeoJSON')
            else:
                pass

if __name__ == "__main__":
    av_cid = sys.argv[1]
    run(av_cid)