import google, ee

def get_region_geometry(country_name, admin_level):
    region_geometry = ee.Feature(ee.FeatureCollection("FAO/GAUL/2015/level{}".format(admin_level)).filter(
        ee.Filter.eq('ADM{}_NAME'.format(admin_level), country_name)).first()).geometry()
    return region_geometry

def get_ntl_mean_region(region_name, region_admin_level, start_date, end_date):
    """
        Parameters
        ----------
        region_name: str
            Official name of the region like Turkey.
        region_admin_level: int
            Administritative level of the region. For example a country would be level 0, province level 1.
        start_date : str
            starting date in the format 'YYYY-MM-DD'.
        end_date : str
             ending date in the format 'YYYY-MM-DD'.
             
        Returns
        ---------
        NTL value for all the pixels in the region aggregated over the year.
    """
    
    # define ImageCollection id
    viirs_rc_id = "NOAA/VIIRS/DNB/MONTHLY_V1/VCMCFG"

    # create an ee object for our image
    viirs_collection = ee.ImageCollection(viirs_rc_id).filterDate(start_date, end_date)
    # view ImageCollection contents
    viirs_collection.getInfo()
    
    # Sums the value for all the months. Each image in the collection is the aggregation of NTL of daily
    # values for a month. So summing over all the 12 months will give us the aggregation for the
    # complete year.
    viirs_img = viirs_collection.reduce(ee.Reducer.sum())
    visrange = {
      min: 0,
      max: 60
    }
    
    # Getting the region geometry
    region_geometry = get_region_geometry(region_name, region_admin_level)
    
    # Reduce the region. The region parameter is the Feature geometry.
    ntl_mean = viirs_img.reduceRegion(reducer=ee.Reducer.sum(), geometry=region_geometry, scale=30, 
                                      maxPixels=1e12)

    return ntl_mean
    

if __name__ == "__main__":
    # ee.Authenticate()
    credentials, project_id = google.auth.default()

    ee.Initialize(credentials, project="thesis-ntl-ankursatya")

    # Check for the working of Google EE API
    print(ee.Image("NASA/NASADEM_HGT/001").get("title").getInfo())

    region_name = "Turkey"
    region_admin_level = 0
    start_date = '2013-01-01'
    end_date = '2013-03-01'

    ntl_mean = get_ntl_mean_region(region_name, region_admin_level, start_date, end_date)
    print(ntl_mean.getInfo()["avg_rad_sum"])
    print(ntl_mean.getInfo()["cf_cvg_sum"])