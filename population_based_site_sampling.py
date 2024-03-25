import numpy as np 
from tqdm import tqdm

class PopulationBasedSampling:
    """A class to sample sites based on population distribution and minium distance between the sites."""

    def __init__(self, pop_array, min_r=2, num_sites=10, k=40, invalid_points=set([]), seed=10):
        np.random.seed(seed)
        
        self.pop_array = pop_array
        self.pop_array[np.where(self.pop_array< 0.0) ] = 0.0
        self.height, self.width = pop_array.shape
        self.min_r = min_r
        self.num_sites = num_sites
        self.k = k #number of candidates
        self.invalid_points = invalid_points
        self.sites = set([])

    # def reset(self):
    #     """Resets the cells dictionary."""

    #     # A list of coordinates in the grid of cells
    #     coords_list = [(iy, ix) for iy in range(self.height)
    #                             for ix in range(self.width)]
    #     # Initilalize the dictionary of cells: each key is a cell's coordinates
    #     # the corresponding value is the index of that cell's point's
    #     # coordinates in the samples list (or None if the cell is empty).
    #     self.cells = {coords: None for coords in coords_list}

    def get_weights_of_points(self):
        self.coords_list = []
        self.total_population = 0.0
        self.weights = []

        for iy in range(self.height):
            for ix in range(self.width):
                if (iy, ix) not in self.invalid_points:
                    self.coords_list.append((iy, ix))
                    self.total_population += self.pop_array[(iy, ix)]

        self.weights = [self.pop_array[coord]/self.total_population for coord in self.coords_list]


    def is_point_valid(self, pt):
        """Is pt a valid point to emit as a sample?

        It must be no closer than r from any other point: check the cells in
        its immediate neighbourhood.

        """
        if pt in self.sites:
            return False
        
        for site in self.sites:
            if np.sqrt((site[0] - pt[0])**2 + (site[1] - pt[1])**2) < self.min_r:
                return False

        return True

    def get_point(self):
        """Get a point from the whole array based on population distribution and it should not be closer 
        than min_r from any other chosen site.
        
        """
        idxs = np.random.choice(np.arange(0, len(self.coords_list)), p=self.weights, size=self.k, replace=False)
        for idx in idxs:
            pt = self.coords_list[idx]
            if self.is_point_valid(pt):
                return pt
        return False


    def sample(self):
        self.get_weights_of_points()

        for _ in range(self.num_sites):
            new_pt = self.get_point()

            if new_pt:
                self.sites.add(new_pt)
        
        if len(self.sites) < self.num_sites:
            print("Could not generate the required num of sites: {} for the given min radius: {}".format(self.num_sites, self.min_r))

        return self.sites