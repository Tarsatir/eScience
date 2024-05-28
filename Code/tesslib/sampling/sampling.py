#Sample generation
import numpy as np

def get_sample_configs(sampling_config):
    if sampling_config["constructive_sampling"] == True :
        #generate sampling configs
        print("reached")
        sample_configs= generate_sample_configs(sampling_config["sampling_matrix"])
    else:
        #read sampling configs from file
        sample_configs = read_sample_configs(sampling_config["sampling_file_path"])
    return sample_configs

def generate_sample_configs(sampling_matrix):
    #combinatorically create sample configurations from sampling values provided
    sample_configs = []
    idcount = 0
    for region in sampling_matrix["regions"]:
        for mode in sampling_matrix["modes"]:
            for fraction in sampling_matrix["point_fractions"]:
                for min_distance in sampling_matrix["min_distances"]:
                    idcount+=1
                    npoints=np.ceil(sampling_matrix["max_points"]*fraction)
                    if mode == "grid":
                        ensemble_size=1
                    else:
                        ensemble_size=sampling_matrix["number_ensemble_members"]
                    sample_config = {"id":idcount, "country_codes":region, "mode":mode, "number_points":npoints, "min_distance":min_distance, "ensemble_size":ensemble_size}
                    sample_configs.append(sample_config)
    return sample_configs

def read_sample_configs(sampling_file_path):
    #TODO
    raise NotImplementedError('Reading of sample config from file not yet implemented')