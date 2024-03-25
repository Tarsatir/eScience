from glob import glob
import shutil
import numpy as np
import os
import time
import socket
import random
import argparse
import pickle
from tqdm import tqdm
from selenium import webdriver
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException, StaleElementReferenceException
from multiprocessing import Process
from population import *
import json


def get_driver(headless=True):
    hostname = socket.gethostname()

    options = Options()
    if headless:
        options.add_argument("--headless")

    options.set_preference('browser.helperApps.neverAsk.saveToDisk', 'image/tiff')
    options.set_preference('browser.download.useDownloadDir', True)
    options.set_preference("browser.download.folderList",2)

    if hostname == "ankur":
        options.set_preference('browser.download.dir', r"/home/ankur/Ankur/CLS/thesis/data/earth_observation_group/new_data")
        driver = webdriver.Firefox(options=options)
    elif hostname in ["int5", "int6"]:
        options.set_preference('browser.download.dir', r"/home/asatya/master-thesis/data/earth_observation_group/new_data")
        options.binary_location = r"/home/asatya/firefox/firefox"
        service = Service(executable_path="/home/asatya/geckodriver")
        driver = webdriver.Firefox(options=options, service=service)
    return driver

def login(driver):
    with open("credentials.json", "w") as f:
        credentials = json.load(f)

    username = credentials["username"]
    password = credentials["password"]

    driver.find_element(By.ID, "username").send_keys(username)
    driver.find_element(By.ID, "password").send_keys(password)
    driver.find_element(By.ID, "kc-login").click()


def get_all_links_from_page(driver):
    table = driver.find_element(By.TAG_NAME, "tbody")
    href_elems = []
    for row in table.find_elements(By.TAG_NAME, "tr"):
        if row.get_attribute("class") in ["even", "odd"]:
            for col in row.find_elements(By.TAG_NAME, "td"):
                if col.get_attribute("class") == "indexcolname":
                    href_elem = col.find_elements(By.TAG_NAME, "a")[0]
                    href_elems.append(href_elem)

    return href_elems

def get_filenames(base_url, ntl_type):
    ## Check if filenames file already exist
    save_filename = "./data/monthly_ntl_filenames_{}.pkl".format(ntl_type)
    if os.path.exists(save_filename):
        with open(save_filename, "rb") as f:
            filenames = pickle.load(f)
    else:
        filenames = {}
        driver = get_driver()
        if ntl_type == "dmsp_ols":
            for year in tqdm(np.arange(1992, 2012+1)):
                url = base_url+str(year)+"/"
                driver.get(url)

                href_elems = get_all_links_from_page(driver)

                filtered_href_elems = [elem for elem in href_elems if "avg_vis.tif" in elem.text]
                filtered_href_texts = [elem.text for elem in filtered_href_elems]
                filenames[year] = filtered_href_texts

        elif ntl_type == "viirs":
            month_str = lambda month : str(month) if month >= 10 else "0"+str(month)
            for year in tqdm(np.arange(2013, 2021+1)):
                url = base_url+str(year)+"/"
                for month in np.arange(1, 12+1):
                    month_url = url+str(year)+month_str(month)+"/vcmcfg/"
                    print(month_url)
                    driver.get(month_url)

                    href_elems = get_all_links_from_page(driver)

                    selected_file = None
                    for elem in href_elems:
                        if "avg_rade9h.tif" in elem.text and ".gz" not in elem.text:
                            selected_file = elem.text
                            break

                    if year not in filenames:
                        filenames[year] = [selected_file]
                    else:
                        filenames[year].append(selected_file)

        with open(save_filename, "wb") as f:
            pickle.dump(filenames, f)
    
    return filenames
    

def get_missing_filenames(ntl_type, year_files):
    ## Checks the folder(./data/earth_observation_group/new_data) for filenames that haven't been downloaded yet and returns the missing files.
    missing_filenames = {}
    missing_filepath = "./data/missing_filenames_{}.pkl".format(ntl_type)
    if os.path.isfile(missing_filepath):
        with open(missing_filepath, "rb") as f:
            missing_filenames = pickle.load(f)

    year_range = np.arange(1992, 2012+1) if ntl_type == "dmsp_ols" else np.arange(2013, 2021+1)
    
    done_filenames = {}
    for file in glob("./data/earth_observation_group/new_data/*"):
        file = file.split("/")[-1]
        if ntl_type == "dmsp_ols":
            year = int(file.split(".")[0].split("_")[1][0:4])
        else:
            year = int(file.split(".")[0].split("_")[2][0:4])

        if year not in year_range:
            continue

        if year not in done_filenames:
            done_filenames[year] = [file]
        else:
            done_filenames[year].append(file)

    for year in year_range:
        if year not in done_filenames:
            if year not in missing_filenames:
                missing_filenames[year] = year_files[year]
        else:
            if year not in missing_filenames:
                missing_filenames[year] = list(set(year_files[year]) - set(done_filenames[year]))
            else:
                missing_filenames[year] = list(set(missing_filenames[year]) - set(done_filenames[year]))

    with open("./data/missing_filenames_{}.pkl".format(ntl_type), "wb") as f:
        pickle.dump(missing_filenames, f)

    return missing_filenames


def download_ntl_monthly_dmsp_ols(base_url, year, missing_filenames):
    if len(missing_filenames) ==  0:
        print("year {} already done.".format(year))
        return

    url = base_url+str(year)+"/"

    driver = get_driver()
    driver.get(url)

    for i, elem_text in enumerate(missing_filenames):
        elem = driver.find_element(By.LINK_TEXT, elem_text)
        elem.click()

        if is_login_needed(driver):
            login(driver)
            time.sleep(2.0)
            driver.get(url)
        else:
            time.sleep(2.0)

    while not status_check(year, "dmsp_ols"):
        time.sleep(10.0)

    driver.quit()

def download_ntl_monthly_viirs(base_url, year, missing_filenames):
    if len(missing_filenames) ==  0:
        print("year {} already done.".format(year))
        return

    url = base_url+str(year)+"/"
    driver = get_driver()

    for elem_text in tqdm(missing_filenames):
        try:
            month_str = elem_text.split("_")[2][4:6]
            month_url = url+str(year)+month_str+"/vcmcfg/"
            driver.get(month_url)
            elem = driver.find_element(By.LINK_TEXT, elem_text)
            elem.click()
        except Exception as e:
            print("Error: {} while clicking the following url: {}".format(e, month_url))
            continue

        if is_login_needed(driver):
            login(driver)
            time.sleep(2.0)
            driver.get(month_url)
        else:
            time.sleep(2.0)
    
    while not status_check(year, "viirs"):
        time.sleep(10.0)

    driver.quit()


def status_check(year, ntl_type):
    files_location = "./data/earth_observation_group/new_data/"
    running_downloads = [file for file in glob(files_location+"*") if ".part" in file]
    if len(running_downloads) > 0:
        this_year_file_count = 0
        for file in running_downloads:
            if ntl_type == "dmsp_ols":
                running_year = int(file.split("/")[-1].split(".")[0].split("_")[1][0:4])
            elif ntl_type == "viirs":
                running_year = int(file.split("/")[-1].split(".")[0].split("_")[2][0:4])

            if running_year ==  year:
                this_year_file_count += 1
        
        if this_year_file_count >0 :
            print("{} downloads remaining for year {}".format(this_year_file_count, year))
            print(" ")
            return False
        else:
            print("Download for year {} done.".format(year))
            print(" ")
            return True
    else:
        print("Download for all years done.")
        print(" ")
        return True


def is_login_needed(driver):
    try: 
        driver.find_element(By.ID, "username")
        return True
    except NoSuchElementException as e:
        return False


def rearrange_data(ntl_type):
    ## Moves the file from ./data/earth_observation_group/new_data to appropriate(relevant year and month) folders.
    month_str = lambda month : str(month) if month >= 10 else "0"+str(month)
    ## check if directories for each year and corresponding months exists. If not, create them.
    base_path = "./data/earth_observation_group/monthly/"
    for year in np.arange(1992, 2021+1):
        year_path = base_path+str(year)
        if not os.path.isdir(year_path):
            os.mkdir(year_path)
            if year == 1992:
                for month in np.arange(4, 12+1):
                    os.mkdir(year_path+"/"+month_str(month))
            else:
                for month in np.arange(1, 12+1):
                    os.mkdir(year_path+"/"+month_str(month))
        else:
            if year == 1992:
                for month in np.arange(4, 12+1):
                    month_path = year_path+"/"+month_str(month)
                    if not os.path.isdir(month_path):
                        os.mkdir(month_path)
            else:
                for month in np.arange(1, 12+1):
                    month_path = year_path+"/"+month_str(month)
                    if not os.path.isdir(month_path):
                        os.mkdir(month_path)

    ## Moving files to appropriate months folder from the folder where all downloads were tmp stored.
    tmp_index = 1 if ntl_type == "dmsp_ols" else 2
    for file in glob("./data/earth_observation_group/new_data/*"):
        tmp = file.split("/")[-1].split(".")[0].split("_")[tmp_index]
        year, month = tmp[0:4], tmp[4:6]
        dst = base_path+year+"/"+month+"/"+file.split("/")[-1]
        os.rename(file, dst)

def delete_ntl_files(ntl_type):
    base_path = "./data/earth_observation_group/monthly/"
    year_range = np.arange(1992, 2012+1) if ntl_type == "dmsp_ols" else np.arange(2013, 2021+1)
    for year in year_range:
        for month_folder in glob(base_path+str(year)+"/*"):
            for file in glob(month_folder+"/*"):
                if ".tif" in file:
                    os.remove(file)

def combine_ntl_data_in_tif(years=None):
    ## Combines different satellites' data for DMSP-OLS.
    if years is None:
        years = np.arange(1992, 2013, 1)
        
        for year in years:
            for month_folder in glob("./data/earth_observation_group/monthly/{}/*".format(year)):
                tif_files = [file for file in glob(month_folder+"/*") if ".tif" and "cloud" in file]
                
                if len(tif_files) == 2:
                    print("Combining for year: ", year)
                    first_tif = rasterio.open(tif_files[0])
                    tif_meta = first_tif.meta.copy()
                    
                    first_tif = first_tif.read([1])
                    second_tif = rasterio.open(tif_files[1]).read([1])
                    
                    avg_tif = (first_tif + second_tif)/2.0
                    
                    
                    just_tif_name = "combined.global.avg_vis.tif"
                    new_tif_name = "/".join(tif_files[0].split("/")[:-1])+"/"+just_tif_name

                    with rasterio.open(new_tif_name, "w", **tif_meta) as dst:
                        dst.write(avg_tif)
                        
                    ## Relocate different satellite files to a new folder
                    combine_folder_path = month_folder+"/various_satellite_data"
                    if not os.path.isdir(combine_folder_path):
                        os.mkdir(combine_folder_path)
                        
                    for file in tif_files:
                        dst = combine_folder_path + "/" + file.split("/")[-1]
                        os.rename(file, dst)
                    
            
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Arguments for NTL yearly aggregation')
    parser.add_argument('--n_procs', required=False, type=int, help='num of processes')
    parser.add_argument('--ntl_type', required=True, type=str, help='type of NTL data(dmsp_ols or viirs)')
    parser.add_argument('--year', required=False, type=int, help='specific year for which to download data')
    parser.add_argument('--rearrange_data', required=False, type=int, help='1 to just rearrange the downloaded files.')
    parser.add_argument('--delete_files', required=False, type=int, help="1 to delete ntl files")
    parser.add_argument('--crop_ntl', required=False, type=int, help="1 to crop ntl files to only include countries of interest.")
    args = parser.parse_args()

    parser.print_help()
    print("_"*100)
    print()
    print("credentials.json uses credentials to login and download data from https://eogdata.mines.edu/products/ . Enter your credentials in this file to be able to download NTL data. Then remove the exit() command in the code after this print statement.")
    exit(0)

    args.rearrange_data = True if args.rearrange_data == 1 else False
    args.delete_files = True if args.delete_files == 1 else False

    base_url = {
        "dmsp_ols":  "https://eogdata.mines.edu/wwwdata/dmsp/monthly_composites/by_year/",
        "viirs": "https://eogdata.mines.edu/nighttime_light/monthly_notile/v10/"
    }

    ## Getting only filenames for all the years.
    filenames = get_filenames(base_url[args.ntl_type], args.ntl_type)

    missing_filenames = get_missing_filenames(args.ntl_type, filenames)

    [print("{} files remaining for year {}".format(len(val), key)) for key, val in missing_filenames.items()]

    if args.rearrange_data is not None and args.rearrange_data:
        print("rearranging data ...")
        rearrange_data(args.ntl_type)
        exit(0)

    if args.delete_files:
        print("deleting NTL files ...")
        delete_ntl_files(args.ntl_type)
        exit(0)

    args.n_procs = 1 if args.n_procs is None else args.n_procs

    num_procs = args.n_procs
    if args.ntl_type == "dmsp_ols":
        year_range = np.arange(1992, 2012+1, num_procs)
        target_func = download_ntl_monthly_dmsp_ols
    elif args.ntl_type == "viirs":
        year_range = np.arange(2013, 2021+1, num_procs)
        target_func = download_ntl_monthly_viirs

    for start_year in year_range:
        if args.year is not None:
            if start_year != args.year:
                continue
                
        if args.ntl_type == "dmsp_ols":
            end_year = min(start_year+num_procs, 2012+1)
        elif args.ntl_type == "viirs":
            end_year = min(start_year+num_procs, 2021+1)

        all_processes = []

        for year in np.arange(start_year, end_year, 1):
            process = Process(target=target_func, args=(base_url[args.ntl_type], year, missing_filenames[year]))
            all_processes.append(process)

        for process in all_processes:
            process.start()
        
        for process in all_processes:
            process.join()

        if args.crop_ntl == 1:
            pop_tif_nodata = -100.0

            cls_population = Population(stitch_images=False, pop_tif_nodata=pop_tif_nodata)
            union_geom = cls_population.get_combined_geometry()

            is_viirs = True if args.ntl_type == "viirs" else False
            
            pop_tif = rasterio.open("./data/population/count/1995/population_count.tif")

            for ntl_file in glob("./data/earth_observation_group/new_data/*"):
                if is_viirs:
                    tmp_year = int(ntl_file.split("/")[-1].split("_")[2][0:4])
                else:
                    tmp_year = int(ntl_file.split("/")[-1].split("_")[1][0:4])

                if tmp_year != start_year:
                    continue

                ntl_tif = rasterio.open(ntl_file)

                filtered_imgs, transforms = cls_population.get_filtered_pop_and_ntl(pop_tif, ntl_tif, union_geom, is_viirs)

                out_meta = ntl_tif.meta.copy()
                out_meta.update(
                    {
                        "height": filtered_imgs[2].shape[0],
                        "width": filtered_imgs[2].shape[1],
                        "transform": transforms[2]
                    }
                )

                masked_image = np.expand_dims(filtered_imgs[2], axis=0)

                with rasterio.open(ntl_file, "w", **out_meta) as dst:
                    dst.write(filtered_imgs[2], 1)

            print("Cropping of NTL Tiff done.")
            print()


        
