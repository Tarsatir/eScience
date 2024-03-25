import os
import numpy as np
import argparse
from tqdm import tqdm
import rasterio
from rasterio.plot import show
from rasterio.mask import mask
from rasterio.merge import merge
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon, mapping, Point
from glob import glob
from skimage.measure import block_reduce
from rasterio.io import MemoryFile



class Population:
    def __init__(self, stitch_images=False, pop_tif_nodata=-100.0):
        self.country_codes = ["BDI", "KEN", "RWA", "UGA", "TZA"]
        self.country_geometry_dict_filtered = self.get_geometry_dict_filtered()
        self.coords_to_code_file = "../data/coords_to_country.npy"

        self.pop_tif_nodata = pop_tif_nodata
        if stitch_images:
            self.stitch_images()


    def stitch_images(self):
        def sort_lambda(filename):
            tmp = filename.split("/")[-1].split(".")[0].split("_")[-2:]
            row_num, col_num = int(tmp[0][1:]), int(tmp[1][1:])
            return row_num, col_num

        base_path = "./data/population/count/"
        for year_folder in glob(base_path+"*"):
            print("Stitching for year {}".format(year_folder.split("/")[-1]))
            year_files = []
            for cell_folder in glob(year_folder+"/*"):
                if not os.path.isdir(cell_folder):
                    continue
                
                for file in glob(cell_folder+"/*"):
                    if ".tif" in file:
                        year_files.append(file)
            year_files = sorted(year_files, key=sort_lambda)

            stitched_array = merge([rasterio.open(file) for file in year_files])
            
            out_meta = rasterio.open(year_files[0]).meta.copy()
            out_meta.update(
                {
                    "height": stitched_array[0].shape[1],
                    "width": stitched_array[0].shape[2],
                    "transform": stitched_array[1]
                }
            )

            with rasterio.open(year_folder+"/population_count.tif", "w", **out_meta) as f:
                f.write(stitched_array[0])

    def get_geometry_dict(self):
        all_country_shapefile = gpd.read_file("../data/country_geometry/all_country/ne_10m_admin_0_countries.shp")

        # # extract the geometry in GeoJSON format
        all_country_geometry = all_country_shapefile.geometry.values # list of shapely geometries
        country_names = list(all_country_shapefile.SOV_A3)
        country_geometry_dict = {country_names[i]: all_country_geometry[i] for i in range(len(all_country_geometry))}

        return country_geometry_dict

    def get_geometry_dict_filtered(self):
        country_geometry_dict = self.get_geometry_dict()
        # filtered_dict = {key: country_geometry_dict[key] for key in self.country_codes}
        filtered_dict = {}

        for country_code in self.country_codes:
            country_geometry = country_geometry_dict[country_code]
    
            ## This is to remove small islands away from the mainland
            if isinstance(country_geometry, MultiPolygon):
                num_geoms = len(country_geometry_dict[country_code].geoms)
                geoms_area = [country_geometry.geoms[i].area for i in range(num_geoms)]
                country_geometry = country_geometry.geoms[geoms_area.index(max(geoms_area))]

            filtered_dict[country_code] = country_geometry

        return filtered_dict

    
    def get_combined_geometry(self):
        country_geometry_dict = self.get_geometry_dict()

        union_geom = None
        for country_code in self.country_codes:
            country_geometry = country_geometry_dict[country_code]
            
            ## This is to remove small islands away from the mainland
            if isinstance(country_geometry, MultiPolygon):
                num_geoms = len(country_geometry_dict[country_code].geoms)
                geoms_area = [country_geometry.geoms[i].area for i in range(num_geoms)]
                country_geometry = country_geometry.geoms[geoms_area.index(max(geoms_area))]
            
            if union_geom is None:
                union_geom = country_geometry
            else:
                union_geom = union_geom.union(country_geometry)
        return union_geom


    def get_bounding_geom_tif(self, tif_data):
        ## the input could either be a tif data or an instance of shapely.geometry.polygon.Polygon
        left, bottom, right, top = tif_data.bounds
        lon_point_list = [left, left, right, right]
        lat_point_list = [bottom, top, top, bottom]

        return Polygon(zip(lon_point_list, lat_point_list))

    def filter_out_tif(self, tif_data, geometry, nodata_value, ntl_data=False, crop=True):
        polygon_mapping = mapping(geometry)
        out_image, out_image_transform = mask(tif_data, [polygon_mapping], crop=crop, nodata=nodata_value)
        out_image = np.squeeze(out_image)

        ## Removing the first and last columns if crop is true since cropping results in nodata values for these columns.
        if crop:
            if ntl_data:
                out_image = out_image[1:, 1:]
            else:
                out_image = out_image[:, 1:-1]

        return out_image, out_image_transform

    def reduce_resolution(self, data, decrease_factor):
        return block_reduce(data, block_size=(decrease_factor, decrease_factor), func=np.sum)

    def filter_ntl_based_on_pop(self, ntl_img, pop_img, pop_img_cropped_shape, pop_tif_nodata):
        return np.reshape(ntl_img[np.where(np.squeeze(pop_img) != pop_tif_nodata)], pop_img_cropped_shape)

    def get_filtered_pop_and_ntl(self, pop_tif, ntl_tif, union_geom, is_viirs, filter_ntl=True):
        ##Filtering the countries data from the stitched population tif(stitched population tif has more data then needed)
        #Step1: Filtering the rectangular bounding geometry of the union of all the countries from the population and NTL tifs.
        #Step1.1: Filtering the pop data but does not crop the image. It is useful for creating a mask of pixels which belong to the bounding geometry
        pop_out_image, pop_out_image_transform = self.filter_out_tif(pop_tif, self.get_bounding_geom_tif(union_geom), self.pop_tif_nodata, crop=False)
        # print(pop_out_image.shape)

        #Step1.2: Filtering the pop data and also crop the image. This will be the new region of interest which will be further filtered.
        pop_out_image_cropped, pop_out_image_cropped_transform = self.filter_out_tif(pop_tif, self.get_bounding_geom_tif(union_geom), self.pop_tif_nodata)
        # print(pop_out_image_cropped.shape)

        if filter_ntl:
            #Step2.1: Filtering the geometry of population tif from the whole of NTL data(which takes a snapshot of the whole world)
            ntl_out_image, ntl_out_image_transform = self.filter_out_tif(ntl_tif, self.get_bounding_geom_tif(pop_tif), ntl_tif.nodata, ntl_data=True)
        else:
            ntl_out_image, ntl_out_image_transform = ntl_tif.read(), ntl_tif.transform
            ntl_out_image = np.squeeze(ntl_out_image)

        #Step2.2: Reducing the resolution of the NTL filtered data since for VIIRS NTL data the resolution is twice of the population data.
        if is_viirs and filter_ntl:
            ntl_out_image = self.reduce_resolution(ntl_out_image, decrease_factor=2)
            ntl_out_image_transform = ntl_tif.transform * ntl_tif.transform.scale(
                (ntl_tif.width / ntl_out_image.shape[-1]),
                (ntl_tif.height / ntl_out_image.shape[-2])
                )
            # print(ntl_out_image.shape)

        if filter_ntl:
            #Step2.3: Filtering further to get NTL data for the rectangular bounding geometry of the union of all the countries.
            ntl_out_image = self.filter_ntl_based_on_pop(ntl_out_image, pop_out_image, pop_out_image_cropped.shape, self.pop_tif_nodata)

        return (pop_out_image, pop_out_image_cropped, ntl_out_image), (pop_out_image_transform, pop_out_image_cropped_transform, ntl_out_image_transform)

    
    def get_geocoordinates(self, tif_meta, masked_image, masked_image_transform):
        out_meta = tif_meta.copy()
        out_meta.update(
            {
                "height": masked_image.shape[0],
                "width": masked_image.shape[1],
                "transform": masked_image_transform
            }
        )

        masked_image_copy = np.expand_dims(masked_image, axis=0)
        # print(masked_image_copy.shape)

        with MemoryFile() as memfile:
            with memfile.open(**out_meta) as dataset:
                dataset.write(masked_image_copy)
                
                height = masked_image_copy.shape[1]
                width = masked_image_copy.shape[2]
                cols, rows = np.meshgrid(np.arange(width), np.arange(height))
                xs, ys= rasterio.transform.xy(dataset.transform, rows, cols)
                lons= np.array(xs)
                lats = np.array(ys)

        return lons, lats

        #         tmp_polygon = Polygon(zip(np.reshape(lons, lons.shape[0]*lons.shape[1]), 
        #                                  np.reshape(lats, lats.shape[0]*lats.shape[1])))
            
                
        #     with memfile.open() as dataset:
        #         new_tmp_data = dataset.read()
        

    def get_country_code_for_tif(self, coord):
        for code, geometry in self.country_geometry_dict_filtered.items():
            point = Point(coord[0], coord[1])
            if geometry.contains(point):
                return code


    def get_country_code_for_tif_all(self, tif_data_lons, tif_data_lats, save=True):
        if os.path.exists(self.coords_to_code_file):
            print("Country codes for tif found.")
            with open(self.coords_to_code_file, "rb") as f:
                return np.load(f, allow_pickle=True)

        country_codes = []

        for nrow in range(tif_data_lons.shape[0]):
            new_data = []
            for ncol in range(tif_data_lats.shape[1]):
                lon, lat = tif_data_lons[nrow, ncol], tif_data_lats[nrow, ncol]
                new_data.append(self.get_country_code_for_tif([lon, lat]))
            country_codes.append(new_data)

        country_codes = np.array(country_codes)

        if save:
            with open(self.coords_to_code_file, "wb") as f:
                np.save(f, country_codes)
        
        return country_codes