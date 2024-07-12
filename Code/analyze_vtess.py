# Import relevant packages
import numpy as np
import pandas as pd
import random
#import PIL
#import rasterio
#import shapefile
#import geopandas as gpd
#import matplotlib.pyplot as plt
#from PIL import Image
#from pyproj import Transformer
#from rasterio import features
#from rasterio.enums import Resampling
#from rasterio.mask import mask, raster_geometry_mask
#from rasterio.plot import show
#from shapely.geometry import Polygon, MultiPolygon, mapping
#from skimage.measure import block_reduce
#from matplotlib.collections import PatchCollection
#from matplotlib.colors import Normalize
#from matplotlib.colors import LogNorm
#from bridson import poisson_disc_samples
#import matplotlib.patches as patches
#import scienceplots 
#import cpnet
#import os 
import Tesselation as tess
import inference as infce
#import ipykernel
import json
import  tesslib as ts
import os

# os.environ['JAVA_HOME'] = '/opt/homebrew/opt/openjdk/libexec/openjdk.jdk/Contents/Home'
# os.environ['PATH'] = os.environ['JAVA_HOME'] + '/bin:' + os.environ['PATH']
#TODO What package needs this? (MODULE LOAD/AVAIL for Java on snellius)

import igraph as ig



def analyze_network(voronoi_geojson_path, points_geojson_path, output_path):
    data_array = infce.create_dataframe_from_voronoi(voronoi_geojson_path)
    te_matrix, p_value_matrix = infce.compute_TE_significance(data_array)
    te_matrix = infce.correct_TE_matrix(te_matrix, p_value_matrix, save=False)
    attributes = infce.measure_attributes_of_graph(te_matrix)

    #add core and periphery to the attributes
    attributes2 = infce.classify_core_periphery(te_matrix)



    #geojson_path = '/Users/mengeshi/Documents/GitHub/eScience/Data_summary/Tesselations/tessellation_points_1.geojson'
    points_geojson_data = json.load(open(points_geojson_path))
    new_points_geojson_data = infce.assign_attributes_to_geojson(attributes, geojson_data=points_geojson_data)
    new_points_geojson_data = infce.assign_attributes_to_geojson(attributes2, geojson_data=new_points_geojson_data)
    #save new data file using .tofile
    # Convert updated GeoJSON to GeoDataFrame
    gdf = gpd.GeoDataFrame.from_features(new_points_geojson_data['features'])  
    gdf.set_crs(epsg=4326, inplace=True)
    # Save the new GeoDataFrame to a file
    gdf.to_file(output_path, driver='GeoJSON')




def run(av_cid=1):

    # Read the config file.
    config = ts.load_config('./tesslib/config/config_template.yml')

    output_root_directory=config["output_root_directory"]
    output_directory_name=config["output_directory_name"]
    output_directory_path = os.path.join(output_root_directory, output_directory_name)

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

    for sconfig in sample_configs:
        if sconfig.id == int(av_cid):
            for r in range(sconfig.ensemble_size):
                cid = f'S{sconfig.id}_R{r}'

                voronoi_geojson_name = f"voronoi_{cid}_final.geojson"
                points_geojson_name = f"points_{cid}_final.geojson"  #NOTE: points file name is placeholder and needss to be replaced before/while merging branch
                analysis_output_name =  f"analysis_{cid}.geojson"  #NOTE: analyssis file name is placeholder and needss to be replaced before/while merging branch
                voronoi_geojson_path = os.path.join(output_directory_path,voronoi_geojson_name)
                points_geojson_path = os.path.join(output_directory_name,points_geojson_name)
                analysis_output_path = os.path.join(output_directory_name,analysis_output_name)


                #geojson_path = '../Data_summary/Tesselations/attributed_voronoi_all_years_1.geojson'
                #output_path = '../Data_summary/Tesselations/new_attributed_points.geojson'
                #path/properties_adjacencytxt
                #take file name ffrom source path (!)

                analyze_network(voronoi_geojson_path, points_geojson_path, analysis_output_path)



if __name__ == "__main__":
    av_cid = sys.argv[1]
    run(av_cid)