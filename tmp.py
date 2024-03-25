# import requests

# url = "https://eogdata.mines.edu/nighttime_light/annual/v20/2021/VNL_v2_npp_2021_global_vcmslcfg_c202203152300.average.tif.gz"
# filename = url.split("/")[-1]
# with open(filename, "wb") as f:
#     r = requests.get(url)
#     f.write(r.content)

import gzip
import urllib.request
import rasterio
import numpy as np

def download_file(url):
    out_file = './tmp.tiff'

    username = "satya.ankur@gmail.com"
    password = "music television"
    url += '/'
    values = { 'username': username,'password': password }
    data = urllib.parse.urlencode(values)
    data = data.encode("ascii")

    print(url)
    print(data)

    # Download archive
    try:
        # Read the file inside the .gz archive located at url
        with urllib.request.urlopen(url, data) as response:
            with gzip.GzipFile(fileobj=response) as uncompressed:
                file_content = uncompressed.read()

        print("Done 1")
        # write to file in binary mode 'wb'
        with open(out_file, 'wb') as f:
            f.write(file_content)
            return 0

    except Exception as e:
        print(e)
        return 1

def get_ntl_snapshot(places):
    fp = r'./data/earth_observation_group/annual/2013/VNL_v2_npp_2013_global_vcmcfg_c202101211500.average.tif'
    img = rasterio.open(fp)


    buffer_pixels = 100
    lat_long_dict = {
        "indian_ocean": (-11.78618, 67.47882),
        "equator": (0.0, 0.0),
        "amsterdam": (52.3676, 4.9041),
        "london": (51.5072, -0.1276),
        "new york": (40.7128, -74.006),
        "delhi": (28.7041, 77.1025),
        "abuja": (9.0765, 7.3986)
    }

    img_array = img.read(1)

    for place in places:
        ys, xs = lat_long_dict[place]
        row, col = rasterio.transform.rowcol(img.transform, xs, ys)
        cropped_area = img_array[row-buffer_pixels: row+buffer_pixels, col-buffer_pixels: col+buffer_pixels]

        avg_radiance_val = round(np.average(cropped_area), 2)

        plt.title(key.upper() + ": " + str(avg_radiance_val))
        plt.imshow(value, cmap='pink')
        # plt.imsave("ntl_{}.png".format(place), value, cmap='pink')
        plt.savefig("ntl_{}.png".format(place), bbox_inches='tight')


# url = "https://eogdata.mines.edu/nighttime_light/annual/v20/2021/VNL_v2_npp_2021_global_vcmslcfg_c202203152300.average.tif.gz"
# download_file(url)

if __name__ == "__main__":
    places = ["delhi", "london"]
    get_ntl_snapshot(places)
