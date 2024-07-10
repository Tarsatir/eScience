import rasterio
import geopandas as gpd
import numpy as np
from scipy.spatial import cKDTree

def get_region_geometry(regions,all_regions_file_path='../Data_summary/country_geometry/World_Continents.geojson'):
    all_regions_file=gpd.read_file(all_regions_file_path)
    selected_region  = all_regions_file[all_regions_file["CONTINENT"].isin(regions)]
    region_geometry = selected_region.unary_union
    selected_region_gdf = gpd.GeoDataFrame(geometry=[region_geometry],crs=selected_region.crs)
    return selected_region_gdf



def gdf_from_raster(rasterobject, name=None, fill_value=None, file=False, filter=None, func=None,crs=None ):

    if file:
        with rasterio.open(rasterobject) as src:
            crs=src.crs
            if filter is not None:
                if filter.crs != crs:
                    filter = filter.to_crs(crs)
                raster_data, raster_transform = rasterio.mask.mask(src,filter.geometry,crop=True,nodata=fill_value,filled=True)
            else:
                raster_data = src.read(1,masked=True)
                raster_data.fill_value = fill_value
                raster_data.filled()
    else:
        raster_data = rasterobject

    if func is not None:
        raster_data = func(raster_data)
    
    # Extract non-zero population data points
    coords = [(x, y) for y, row in enumerate(raster_data) 
                for x, val in enumerate(row) if val > 0]
    values = [raster_data[y, x] for x, y in coords]

    # Convert array indices to spatial coordinates and create points
    geometry = [Point(src.xy(y, x)) for x, y in coords]

        # Create a GeoDataFrame
    gdf = gpd.GeoDataFrame({name: values, 'geometry': geometry}, crs=crs)
    return gdf
    
def prob_func(raster_data):
    raster_data = np.power(raster_data,2)
    tot_sum = np.sum(raster_data)
    prob = raster_data/tot_sum
    return prob

"""
def raster_to_gdf(raster_data_object, name='value', fill_value=0, file=False):

    if file:
        with rasterio.open(raster_data_object) as src:
            gdf=_raster_to_gdf(src,fill_value=fill_value)
            return gdf
    else:
        gdf=_raster_to_gdf(raster_data_object,fill_value=fill_value)
        return gdf
    

def _raster_to_gdf(src,fill_value=0):
    raster_data = src.read(1,masked=True)
    raster_data.fill_value=fill_value
    raster_data=raster_data.filled()

    # Ensure there's data to process
    if raster_data.sum() == 0:
        raise ValueError("The sum of the raster values is 0; no valid data.")

    # Extract non-zero population data points
    coords = [(x, y) for y, row in enumerate(raster_data)
              for x, val in enumerate(row) if val > 0]
    values = [raster_data[y, x] for x, y in coords]

    # Convert array indices to spatial coordinates and create points
    geometry = [Point(src.xy(y, x)) for x, y in coords]

    # Create a GeoDataFrame
    gdf = gpd.GeoDataFrame({name: values, 'geometry': geometry}, crs=src.crs)
    return gdf
"""
    
def append_metadata_to_gdf(gdf,metadata):
    gdf.attrs = {"metadata":metadata}  
    return gdf

def filter_gdf_by_geometry(gdf,filter_geom_gdf):
    filtered_gdf = gpd.sjoin(gdf, filter_geom_gdf, predicate='within')
    return filtered_gdf

def poisson_disk_sampling(gdf, mode, weights=None, filter_geom=None, radius=None, n_samples=100):
    if mode=='density':
        points = _poisson_disk_sampling(gdf, weights=weights, radius=radius,n_samples=n_samples)
    elif mode=='random':
        if filter_geom is not None:
            input_gdf = filter_gdf_by_geometry(gdf,filter_geom)
        else:
            input_gdf = gdf
        points = _poisson_disk_sampling(gdf, radius=radius,n_samples=n_samples)
    else:
        raise ValueError(f'mode {mode} not supported by poisson_disk_sampling() ')
    return points


def _poisson_disk_sampling(geo_df, weights=None, radius=None, n_samples=100):
    """
    Perform Poisson disk sampling on a geospatial DataFrame,
    with a minimum radius between points, optionally weighted.

    Parameters:
    radius (float): Minimum distance between samples.
    geo_df (geopandas.GeoDataFrame): GeoDataFrame containing point geometries and their probabilities.
    n_samples (int): Max number of samples to generate, each at least `radius` apart
    weight (string): column name to use as weight

    Returns:
    geopandas.GeoDataFrame: GeoDataFrame of the sampled points.
    """
    # Sample initial points based on weighted probability
    if weights is not None:
        initial_points = geo_df.sample(n=n_samples, weights=weights, replace=False)
    else:
        initial_points = geo_df.sample(n=n_samples,replace=False)
    
    points_array = np.array([[point.x, point.y] for point in initial_points.geometry])
    tree = cKDTree(points_array)
    
    # Filter points based on radius
    valid_points = []
    indices_to_remove = set()

    for i in range(len(points_array)):
        if i in indices_to_remove:
            continue
        # Points within the radius including the point itself
        indices = tree.query_ball_point(points_array[i], radius)
        # Remove the point itself from the set
        indices.remove(i)
        # Update indices to remove with the neighbors, keeping only the first point
        indices_to_remove.update(indices)
        # Add the valid point (non-removed point)
        valid_points.append(initial_points.iloc[i])

    # Convert list of points to GeoDataFrame
    return gpd.GeoDataFrame(geometry=[point.geometry for point in valid_points], crs=geo_df.crs)