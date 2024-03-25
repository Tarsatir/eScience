##Implementation of the voronoi tessellation creation

import numpy as np 
import scipy as sc
from scipy.stats import qmc
from heapq import heappop, heappush, heapify, nsmallest
from geopy.distance import geodesic
from poisson_disc_sampling import *
from population_based_site_sampling import *
from tqdm import tqdm
import pickle
import warnings
from scipy.spatial import cKDTree
import pandas as pd

class VoronoiTessellation:
    def __init__(self, region_array, true_region_mask, num_sites, min_radius, lat_bounds, lon_bounds, seed=10, poisson_based=False):
        self.seed = seed
        self.region_array = region_array
        self.true_region_mask = true_region_mask #stores the mask to show which cells in the region array belong to the countries of interest.

        self.min_radius = min_radius #minimum distance in kms between two sites
        self.lat_bounds = lat_bounds
        self.lon_bounds = lon_bounds
        self.num_sites = num_sites
        self.poisson_based=poisson_based

        self.unit_per_km_lat, self.unit_per_km_lon = self.convert_kms_to_grid_unit()

        # self.total_capacity = np.sum(self.region_array[self.true_region_mask])
        # self.evaluate_per_site_capacity()
        

    def evaluate_per_site_capacity(self):
        self.per_site_capacity = self.total_capacity/self.num_sites

    def convert_kms_to_grid_unit(self):
        grid_shape = self.region_array.shape

        unit_per_km_lat_1 = grid_shape[0]/geodesic((self.lat_bounds[0], self.lon_bounds[0]), (self.lat_bounds[1], self.lon_bounds[0])).km
        unit_per_km_lat_2 = grid_shape[0]/geodesic((self.lat_bounds[0], self.lon_bounds[1]), (self.lat_bounds[1], self.lon_bounds[1])).km
        unit_per_km_lat = (unit_per_km_lat_1 + unit_per_km_lat_2)/2
        
        unit_per_km_lon_1 = grid_shape[1]/geodesic((self.lat_bounds[0], self.lon_bounds[0]), (self.lat_bounds[0], self.lon_bounds[1])).km
        unit_per_km_lon_2 = grid_shape[1]/geodesic((self.lat_bounds[1], self.lon_bounds[0]), (self.lat_bounds[1], self.lon_bounds[1])).km
        unit_per_km_lon = (unit_per_km_lon_1 + unit_per_km_lon_2)/2
        
    
        return unit_per_km_lat, unit_per_km_lon
    
    # converts distance from kilometers to the grid unit.
    def get_distance_in_grid_unit(self, distance):
        return (np.sqrt(np.sum(np.square([distance*self.unit_per_km_lat, distance*self.unit_per_km_lon]))))

    # initialises sites with minimum radius between them.
    def initialise_sites(self):
        with open('./data/coords_to_country.npy', 'rb') as f:
            e = np.load(f, allow_pickle=True)
        
        u_bounds = self.region_array.shape
        radius = self.get_distance_in_grid_unit(self.min_radius)
        print("min radius, in kms: {}, in grid units: {}".format(self.min_radius, radius))

        points_to_ignore = set(tuple(map(tuple, np.argwhere(e == None))))

        if self.poisson_based:
            print("using poisson disk sampling for initialising sites...")
            pd = PoissonDisc(width=u_bounds[1], height=u_bounds[0], r=radius, k=30, n=self.num_sites, invalid_points=points_to_ignore, seed=self.seed)
            sites = pd.sample()
        else:
            print("using population distribution based sampling for initialising sites...")
            pop_based_sampling = PopulationBasedSampling(self.region_array, min_r=radius, num_sites=self.num_sites, k=30, invalid_points=points_to_ignore, seed=self.seed)
            sites = list(pop_based_sampling.sample())

        if len(sites) < self.num_sites:
            print("Num of sites generated is: {} while expectation was: {}.\n Try decreasing either the num of sites or the minimum distance between them.".format(len(sites), self.num_sites))
            self.num_sites = len(sites)
            # self.evaluate_per_site_capacity()

        self.sites = sites

        with open("./data/sites_data.npy", "wb") as f:
            np.save(f, np.array(sites))
        

    # returns the initial configuration of the assignment of points to sites.
    def get_initial_assignment(self):
        # Initialization can be done by assigning each site equal capacity irrespective of the distance from the site.
        masked_region_array = sorted(self.region_array[self.true_region_mask])
        site_to_point_dict = {num: set([]) for num in range(self.num_sites)}
        site_to_capacity = {}
        
        site_num = 0
        site_capacity = 0.0

        for i in range(self.region_array.shape[0]):
            for j in range(self.region_array.shape[1]):
                if not self.true_region_mask[i, j]:
                    continue

                site_capacity += self.region_array[i, j]
                site_to_point_dict[site_num].add((i,j))

                if site_capacity >= self.per_site_capacity and site_num != self.num_sites - 1:
                    site_to_capacity[site_num] = site_capacity
                    site_num += 1
                    site_capacity = 0.0

        self.site_to_point_dict = site_to_point_dict

    def get_distance_between_points(self, pt_1, pt_2, squared=True):
        dist = (pt_1[0]-pt_2[0])**2 + (pt_1[1] - pt_2[1])**2
        if squared:
            return dist
        return np.sqrt(dist)


    def calculate_centroid_sites(self):
        centroid_sites = {}
        for key, item in self.site_to_point_dict.items():
            numerator_x = 0.0
            numerator_y = 0.0
            # total_mass = 0.0
            for point in item:
                # mass = self.region_array[point[0], point[1]]
                # total_mass += mass
                # numerator_x += (mass*point[1])
                # numerator_y += (mass*point[0])
                numerator_x += point[1]
                numerator_y += point[0]
                
            # com_x = np.floor(numerator_x/total_mass)
            # com_y = np.floor(numerator_y/total_mass)

            centroid_x = np.floor(numerator_x/len(item))
            centroid_y = np.floor(numerator_y/len(item))
            
            centroid_sites[key] = [centroid_y, centroid_x]
            
        return centroid_sites

    def relocate_sites_to_centroid(self):
        centroid_sites = self.calculate_centroid_sites()

        for id_, site in enumerate(self.sites):
            if id_ not in centroid_sites:
                continue

            self.sites[id_] = centroid_sites[id_]


    # returns the stable capacity-constrained assignment of points to sites.
    def get_stable_assignment(self):
        ## Current: Overall energy is decreasing. Implement a stability criterion. Then implement moving the sites to centroid and then compare the number of iterations in this new method vs before.
        ## Stability criterion can be build upon the change in log of total energy or something similar. before that check if the way you are using log energy makes sense.
        stable = False

        total_energy = 0.0
        iteration_count = 1

        while not stable:
            print("iteration number: ", iteration_count)
            
            stable = True
            for i in tqdm(range(self.num_sites)):
                for j in range(i+1, self.num_sites):
                    H_i, H_j = [], []
                    # heapify(H_i), heapify(H_j)
                    energy_hi, energy_hj = 0.0, 0.0

                    for point in self.site_to_point_dict[i]:
                        key = self.get_distance_between_points(point, self.sites[i]) - self.get_distance_between_points(point, self.sites[j])
                        #modifying the distance based on the capacity of the point.
                        # key *= self.region_array[point[0], point[1]]
                        # if self.region_array[point[0], point[1]] != 0:
                        #     key/= self.region_array[point[0], point[1]]
                        # else:
                        #     if key != 0.0:
                        #         key = np.sign(key)*np.inf
                                

                        # heappush(H_i, [-1.0*key, point])
                        H_i.append([key, point])
                        energy_hi += key

                    for point in self.site_to_point_dict[j]:
                        key = self.get_distance_between_points(point, self.sites[j]) - self.get_distance_between_points(point, self.sites[i])
                        #modifying the distance based on the capacity of the point.
                        # key *= self.region_array[point[0], point[1]]
                        # if self.region_array[point[0], point[1]] != 0:
                        #     key/= self.region_array[point[0], point[1]]
                        # else:
                        #     if key != 0.0:
                        #         key = np.sign(key)*np.inf


                        # heappush(H_j, [-1.0*key, point])
                        H_j.append([key, point])
                        energy_hj += key
                    
                    H_i = sorted(H_i)
                    H_j = sorted(H_j)

                    if energy_hi > 0:
                        energy_hi = np.log(energy_hi)
                    else:
                        if energy_hi == 0.0:
                            energy_hi = -np.inf
                        energy_hi= -1.0*np.log(-1.0*energy_hi)

                    if energy_hj > 0:
                        energy_hj = np.log(energy_hj)
                    else:
                        if energy_hj == 0.0:
                            energy_hj = -np.inf
                        energy_hj= -1.0*np.log(-1.0*energy_hj)

                    total_energy += energy_hi+energy_hj

                    # print("Energies for site: {} and {} are: {}, {}. Sum: {}".format(i, j, energy_hi, energy_hj, energy_hi+energy_hj))

                    while len(H_i) > 0 and len(H_j) > 0 and (H_i[-1][0] + H_j[-1][0]) > 0:
                    # while len(H_i) > 0 and len(H_j) > 0:
                    #     with warnings.catch_warnings():
                    #         warnings.filterwarnings('error')
                    #         try:
                    #             if (H_i[-1][0] + H_j[-1][0]) > 0:
                    #                 h_i_max, h_j_max = H_i.pop(), H_j.pop()
                    #                 h_i_max_point, h_j_max_point = h_i_max[1], h_j_max[1]


                    #                 self.site_to_point_dict[j].add(h_i_max_point)
                    #                 self.site_to_point_dict[i].add(h_j_max_point)

                    #                 self.site_to_point_dict[i].remove(h_i_max_point)
                    #                 self.site_to_point_dict[j].remove(h_j_max_point)

                    #                 stable = False
                    #             else:
                    #                 break
                    #         except Warning as e:
                    #             # print("Warning: ", e)
                    #             # print(H_i[-1])
                    #             # print(H_j[-1])
                    #             break


                        h_i_max, h_j_max = H_i.pop(), H_j.pop()
                        h_i_max_point, h_j_max_point = h_i_max[1], h_j_max[1]


                        self.site_to_point_dict[j].add(h_i_max_point)
                        self.site_to_point_dict[i].add(h_j_max_point)

                        self.site_to_point_dict[i].remove(h_i_max_point)
                        self.site_to_point_dict[j].remove(h_j_max_point)

                        stable = False

                    # while len(H_i) > 0 and len(H_j) > 0 and (-1.0*(nsmallest(1, H_i)[0][0] + nsmallest(1, H_j)[0][0])) > 0:
                        # h_i_max, h_j_max = heappop(H_i), heappop(H_j)
                        # h_i_max[0] *= -1.0
                        # h_j_max[0] *= -1.0
                        # h_i_max_point, h_j_max_point = h_i_max[1], h_j_max[1]

                        # self.site_to_point_dict[j].add(h_i_max_point)
                        # self.site_to_point_dict[i].add(h_j_max_point)

                        # self.site_to_point_dict[i].remove(h_i_max_point)
                        # self.site_to_point_dict[j].remove(h_j_max_point)

            iteration_count += 1
            print("total_energy: ", total_energy)

            site_dict = {i: site for i, site in enumerate(self.sites)}
            data_save_dict = {
                "sites_location": site_dict,
                "site_to_point": self.site_to_point_dict
                }

            sampling_type = "poisson" if self.poisson_based else "population_based"

            with open("./data/voronoi_tessellations/no-criterion_radius_{}_sites_{}_{}_iteration-count_{}.pkl".format(self.min_radius, self.num_sites, sampling_type, iteration_count-1), "wb") as f:
                pickle.dump(data_save_dict, f)

            
            #Moving sites to the center of mass of the points belonging to that site.
            self.relocate_sites_to_centroid()

            
            # with open("./data/voronoi_tessellations/no-criterion_radius_{}_sites_{}_site.pkl".format(self.min_radius, self.num_sites), "wb") as f:
            #     pickle.dump(site_dict, f)

    def get_voronoi_cells(self):
        self.site_to_point_dict = {num: set([]) for num in range(self.num_sites)}

        ys, xs = np.where(self.true_region_mask == True)
        pts = [[x,y] for x,y in zip(xs, ys)]

        voronoi_kdtree = cKDTree(self.sites)
        point_dist, point_sites = voronoi_kdtree.query(pts)

        for pt, site in zip(pts, point_sites):
            self.site_to_point_dict[site].add((pt[1], pt[0]))

        sites_to_delete = []
        for site, site_pts in self.site_to_point_dict.items():
            if len(site_pts) == 0:
                sites_to_delete.append(site)

        if len(sites_to_delete):
            print("{} sites removed".format(len(sites_to_delete)))

        for site in sites_to_delete:
            del self.site_to_point_dict[site]

        self.relocate_sites_to_centroid()

        # site_dict = {i: site for i, site in enumerate(self.sites)}
        # data_save_dict = {
        #     "sites_location": site_dict,
        #     "site_to_point": self.site_to_point_dict
        #     }

        # with open("./data/tmp.pkl", "wb") as f:
        #     pickle.dump(data_save_dict, f)
            

    def run(self):
        self.initialise_sites()

        ## When using the equal capacity voronoi tessellation
        # self.get_initial_assignment()
        # self.get_stable_assignment()

        ## Otherwise
        self.get_voronoi_cells()
        

        # self.get_voronoi_region_stats()






        