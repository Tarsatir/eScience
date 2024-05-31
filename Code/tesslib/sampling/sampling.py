#Sample generation
import numpy as np

class sample_configuration(object):
    def __init__(self):
        self.id=None
        self.country_codes=None
        self.mode=None
        self.number_points=None
        self.min_distance=None
        self.ensemble_size=None
        self.config_file=None


def get_sample_configs(sampling_config,config_file=None):
    if sampling_config["constructive_sampling"] == True :
        #generate sampling configs
        print("reached")
        sample_configs= generate_sample_configs(sampling_config["sampling_matrix"],config_file=config_file)
    else:
        #read sampling configs from file
        sample_configs = read_sample_configs(sampling_config["sampling_file_path"],config_file=config_file)
    return sample_configs

def generate_sample_configs(sampling_matrix,config_file=None):
    #combinatorically create sample configurations from sampling values provided
    sample_configs = []
    idcount = 0
    for region in sampling_matrix["regions"]:
        for mode in sampling_matrix["modes"]:
            for fraction in sampling_matrix["point_fractions"]:
                for min_distance in sampling_matrix["min_distances"]:
                    idcount+=1
                    npoints=np.ceil(sampling_matrix["max_points"]*fraction).astype('int')
                    if mode == "grid":
                        ensemble_size=1
                    else:
                        ensemble_size=sampling_matrix["number_ensemble_members"]
                    sample_config = sample_configuration()
                    sample_config.id=idcount
                    sample_config.country_codes=region
                    sample_config.mode=mode
                    sample_config.number_points=npoints
                    sample_config.min_distance=min_distance
                    sample_config.ensemble_size=ensemble_size
                    sample_config.config_file=config_file    
                    #sample_config = {"id":idcount, "country_codes":region, "mode":mode, "number_points":npoints, "min_distance":min_distance, "ensemble_size":ensemble_size}
                    sample_configs.append(sample_config)
    return sample_configs

def read_sample_configs(sampling_file_path,config_file=None):
    #TODO
    raise NotImplementedError('Reading of sample config from file not yet implemented')
