import heapq
import numpy as np
from scipy.spatial import Rectangle

from flightsim.world import World
from flightsim.shapes import Cuboid


class OccupancyMap:
    def __init__(self, world=World.empty((0, 2, 0, 2, 0, 2)), resolution=(.1, .1, .1), margin=.2):
        """
        This class creates a 3D voxel occupancy map of the configuration space from a flightsim World object.
        Parameters:
            world, a flightsim World object
            resolution, the discretization of the occupancy grid in x,y,z
            margin, the inflation radius used to create the configuration space (assuming a spherical drone)
        """
        self.world = world
        self.resolution = np.array(resolution)
        self.margin = margin
        self.map = np.array
        self.create_map_from_world()

    def index_to_metric_negative_corner(self, index):
        """
        Return the metric position of the most negative corner of a voxel, given its index in the occupancy grid
        """
        return index*np.array(self.resolution) + self.origin

    def index_to_metric_center(self, index):
        """
        Return the metric position of the center of a voxel, given its index in the occupancy grid
        """
        return self.index_to_metric_negative_corner(index) + self.resolution/2.0

    def metric_to_index(self, metric):
        """
        Returns the index of the voxel containing a metric point.
        Remember that this and index_to_metric and not inverses of each other!
        """
        return np.floor((metric - self.origin)/self.resolution).astype('int')

    def create_map_from_world(self):
        """
        Creates the occupancy grid (self.map) as a boolean numpy array. True is occupied, False is unoccupied.
        This function is called during initialization of the object.
        """
        bounds = self.world.world['bounds']['extents']
        voxel_dimensions_metric = []
        voxel_dimensions_indices = []
        for i in range(3):
            voxel_dimensions_metric.append(abs(bounds[1+i*2]-bounds[i*2]))
            voxel_dimensions_indices.append(int(np.ceil(voxel_dimensions_metric[i]/self.resolution[i])))
            # initialize the map with the correct dimensions as unoccupied
        self.map = np.zeros(voxel_dimensions_indices, dtype=bool)
        self.origin = np.array([bounds[0], bounds[2], bounds[4]])

        # Create Rectangle objects from the obstacles to use for computing the configuration space
        obstacle_rects = []
        if 'blocks' in self.world.world:
            for block in self.world.world['blocks']:
                extent = block['extents']
                obstacle_rects.append(Rectangle([extent[1], extent[3], extent[5]], [extent[0], extent[2], extent[4]]))

        it = np.nditer(self.map, flags=['multi_index'])
        # Iterate through every voxel in the map and check if it is too close to an obstacle. If so, mark occupied
        while not it.finished:
            metric_loc = self.index_to_metric_negative_corner(it.multi_index)
            voxel_rectangle = Rectangle(metric_loc+self.resolution, metric_loc)
            for obstacle in obstacle_rects:
                rect_distance = voxel_rectangle.min_distance_rectangle(obstacle)
                if rect_distance <= self.margin:
                    self.map[it.multi_index] = True
            it.iternext()

    def draw(self, ax):
        """
        Visualize the occupancy grid (mostly for debugging)
        Warning: may be slow with O(10^3) occupied voxels or more
        Parameters:
            ax, an Axes3D object
        """
        self.world.draw_empty_world(ax)
        it = np.nditer(self.map, flags=['multi_index'])
        while not it.finished:
            if self.map[it.multi_index] == True:
                metric_loc = self.index_to_metric_negative_corner(it.multi_index)
                xmin, ymin, zmin = metric_loc
                xmax, ymax, zmax = metric_loc + self.resolution
                c = Cuboid(ax, xmax-xmin, ymax-ymin, zmax-zmin, alpha=0.1, linewidth=1, edgecolors='k', facecolors='b')
                c.transform(position=(xmin, ymin, zmin))
            it.iternext()

    def is_valid_index(self, voxel_index):
        """
        Test if a voxel index is within the map.
        Returns True if it is inside the map, False otherwise.
        """
        for i in range(3):
            if voxel_index[i] >= self.map.shape[i] or voxel_index[i] < 0:
                return False
        return True

    def is_valid_metric(self, metric):
        """
        Test if a metric point is within the world.
        Returns True if it is inside the world, False otherwise.
        """
        bounds = self.world.world['bounds']['extents']
        for i in range(3):
            if metric[i] <= bounds[i*2] or metric[i] >= bounds[i*2+1]:
                return False
        return True

    def is_occupied_index(self, voxel_index):
        """
        Test if a voxel index is occupied.
        Returns True if occupied, False otherwise.
        """
        return self.map[tuple(voxel_index)]

    def is_occupied_metric(self, voxel_metric):
        """
        Test if a metric point is within an occupied voxel.
        Returns True if occupied, False otherwise.
        """
        ind = self.metric_to_index(voxel_metric)
        return self.is_occupied_index(ind)


if __name__ == "__main__":

    from flightsim.axes3ds import Axes3Ds
    import matplotlib.pyplot as plt

    # Create a world object first
    world = World.random_forest(world_dims=(5, 5, 5), tree_width=.1, tree_height=5, num_trees=10)

    # Create a figure
    fig = plt.figure()
    ax = Axes3Ds(fig)
    # Draw the world
    # world.draw(ax)

    # Create an Occupancy map
    oc = OccupancyMap(world, (.2, .2, .5), .1)
    # Draw the occupancy map (may be slow for many voxels; will look weird if plotted on top of a world.draw)
    oc.draw(ax)

    plt.show()
