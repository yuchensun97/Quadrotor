import json
import numpy as np

from flightsim.shapes import Cuboid
from flightsim.numpy_encoding import NumpyJSONEncoder, to_ndarray

def interp_path(path, res):
    cumdist = np.cumsum(np.linalg.norm(np.diff(path, axis=0),axis=1))
    t = np.insert(cumdist,0,0)
    ts = np.arange(0, cumdist[-1], res)
    pts = np.empty((ts.size, 3), dtype=np.float)
    for k in range(3):
        pts[:,k] = np.interp(ts, t, path[:,k])
    return pts

class World(object):

    def __init__(self, world_data):
        """
        Construct World object from data. Instead of using this constructor
        directly, see also class methods 'World.from_file()' for building a
        world from a saved .json file or 'World.grid_forest()' for building a
        world object of a parameterized style.

        Parameters:
            world_data, dict containing keys 'bounds' and 'blocks'
                bounds, dict containing key 'extents'
                    extents, list of [xmin, xmax, ymin, ymax, zmin, zmax]
                blocks, list of dicts containing keys 'extents' and 'color'
                    extents, list of [xmin, xmax, ymin, ymax, zmin, zmax]
                    color, color specification
        """
        self.world = world_data

    @classmethod
    def from_file(cls, filename):
        """
        Read world definition from a .json text file and return World object.

        Parameters:
            filename

        Returns:
            world, World object

        Example use:
            my_world = World.from_file('my_filename.json')
        """
        with open(filename) as file:
            return cls(to_ndarray(json.load(file)))

    def to_file(self, filename):
        """
        Write world definition to a .json text file.

        Parameters:
            filename

        Example use:
            my_word.to_file('my_filename.json')
        """
        with open(filename, 'w') as file:  # TODO check for directory to exist
            file.write(json.dumps(self.world, cls=NumpyJSONEncoder, indent=4))

    def closest_points(self, points):
        """
        For each point, return the closest occupied point in the world and the
        distance to that point. This is appropriate for computing sphere-vs-world
        collisions.

        Input
            points, (N,3)
        Returns
            closest_points, (N,3)
            closest_distances, (N,)
        """

        closest_points = np.empty_like(points)
        closest_distances = np.full(points.shape[0], np.inf)
        p = np.empty_like(points)
        for block in self.world.get('blocks', []):
            # Computation takes advantage of axes-aligned blocks. Note that
            # scipy.spatial.Rectangle can compute this distance, but wouldn't
            # return the point itself.
            r = block['extents']
            for i in range(3):
                p[:, i] = np.clip(points[:, i], r[2*i], r[2*i+1])
            d = np.linalg.norm(points-p, axis=1)
            mask = d < closest_distances
            closest_points[mask, :] = p[mask, :]
            closest_distances[mask] = d[mask]
        return (closest_points, closest_distances)

    def path_collisions(self, path, margin):
        """
        Densely sample the path and check for collisions. Return a boolean mask
        over the samples and the sample points themselves.
        """
        pts = interp_path(path, res=0.001)
        (closest_pts, closest_dist) = self.closest_points(pts)
        collisions = closest_dist < margin
        return (collisions, pts)

    def draw_empty_world(self, ax):
        """
        Draw just the world without any obstacles yet. The boundary is represented with a black line.
        Parameters:
            ax, Axes3D object
        """
        (xmin, xmax, ymin, ymax, zmin, zmax) = self.world['bounds']['extents']

        # Set axes limits all equal to approximate 'axis equal' display.
        x_width = xmax-xmin
        y_width = ymax-ymin
        z_width = zmax-zmin
        width = np.max((x_width, y_width, z_width))
        ax.set_xlim((xmin, xmin+width))
        ax.set_ylim((ymin, ymin+width))
        ax.set_zlim((zmin, zmin+width))
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        c = Cuboid(ax, xmax-xmin, ymax-ymin, zmax-zmin, alpha=0.01, linewidth=1, edgecolors='k')
        c.transform(position=(xmin, ymin, zmin))

    def draw(self, ax):
        """
        Draw world onto existing Axes3D axes and return artists corresponding to the
        blocks.

        Parameters:
            ax, Axes3D object

        Returns:
            block_artists, list of Artists associated with blocks

        Example use:
            my_world.draw(ax)
        """

        self.draw_empty_world(ax)
        # This doesn't look nice because the z-order isn't sophisticated enough.
        # wireframe = Cuboid(ax, xmax-xmin, ymax-ymin, zmax-zmin, alpha=0, linewidth=1, edgecolors='k')
        # wireframe.transform((xmin,ymin,zmin))

        block_artists = []
        for b in self.world.get('blocks', []):
            (xmin, xmax, ymin, ymax, zmin, zmax) = b['extents']
            c = Cuboid(ax, xmax-xmin, ymax-ymin, zmax-zmin, alpha=0.6, linewidth=1, edgecolors='k')
            c.transform(position=(xmin, ymin, zmin))
            block_artists.extend(c.artists)
        return block_artists

    def draw_line(self, ax, points, color=None):
        path_length = np.sum(np.linalg.norm(np.diff(points, axis=0),axis=1))
        pts = interp_path(points, res=path_length/1000)
        ax.scatter(pts[:,0], pts[:,1], pts[:,2], s=3, c=color, edgecolors='none', depthshade=False)

    def draw_points(self, ax, points, color=None):
        ax.scatter(points[:,0], points[:,1], points[:,2], s=16, c=color, edgecolors='none', depthshade=False)

    # The follow class methods are convenience functions for building different
    # kinds of parametric worlds.

    @classmethod
    def empty(cls, extents):
        """
        Return World object for bounded empty space.

        Parameters:
            extents, tuple of (xmin, xmax, ymin, ymax, zmin, zmax)

        Returns:
            world, World object

        Example use:
            my_world = World.empty((xmin, xmax, ymin, ymax, zmin, zmax))
        """
        bounds = {'extents': extents}
        blocks = []
        world_data = {'bounds': bounds, 'blocks': blocks}
        return cls(world_data)

    @classmethod
    def grid_forest(cls, n_rows, n_cols, width, height, spacing):
        """
        Return World object describing a grid forest world parameterized by
        arguments. The boundary extents fit tightly to the included trees.

        Parameters:
            n_rows, rows of trees stacked in the y-direction
            n_cols, columns of trees stacked in the x-direction
            width, weight of square cross section trees
            height, height of trees
            spacing, spacing between centers of rows and columns

        Returns:
            world, World object

        Example use:
            my_world = World.grid_forest(n_rows=4, n_cols=3, width=0.5, height=3.0, spacing=2.0)
        """

        # Bounds are outer boundary for world, which are an implicit obstacle.
        x_max = (n_cols-1)*spacing + width
        y_max = (n_rows-1)*spacing + width
        bounds = {'extents': [0, x_max, 0, y_max, 0, height]}

        # Blocks are obstacles in the environment.
        x_root = spacing * np.arange(n_cols)
        y_root = spacing * np.arange(n_rows)
        blocks = []
        for x in x_root:
            for y in y_root:
                blocks.append({'extents': [x, x+width, y, y+width, 0, height], 'color': [1, 0, 0]})

        world_data = {'bounds': bounds, 'blocks': blocks}
        return cls(world_data)

    @classmethod
    def random_forest(cls, world_dims, tree_width, tree_height, num_trees):
        """
        Return World object describing a random forest world parameterized by
        arguments.

        Parameters:
            world_dims, a tuple of (xmax, ymax, zmax). xmin,ymin, and zmin are set to 0.
            tree_width, weight of square cross section trees
            tree_height, height of trees
            num_trees, number of trees

        Returns:
            world, World object
        """

        # Bounds are outer boundary for world, which are an implicit obstacle.
        bounds = {'extents': [0, world_dims[0], 0, world_dims[1], 0, world_dims[2]]}

        # Blocks are obstacles in the environment.
        xs = np.random.uniform(0, world_dims[0], num_trees)
        ys = np.random.uniform(0, world_dims[1], num_trees)
        pts = np.stack((xs, ys), axis=-1) # min corner location of trees
        w, h = tree_width, tree_height
        blocks = []
        for pt in pts:
            extents = list(np.round([pt[0], pt[0]+w, pt[1], pt[1]+w, 0, h], 2))
            blocks.append({'extents': extents, 'color': [1, 0, 0]})

        world_data = {'bounds': bounds, 'blocks': blocks}
        return cls(world_data)


if __name__ == '__main__':
    from flightsim.axes3ds import Axes3Ds
    import matplotlib.pyplot as plt

    # Test grid_forest world.
    world = World.grid_forest(n_rows=4, n_cols=3, width=0.5, height=3.0, spacing=2.0)
    # Save to file.
    world.to_file('worlds/grid_forest.json')
    # Build a new world from the saved file.
    world = World.from_file('worlds/grid_forest.json') # TODO: Path is brittle.
    # Draw.
    fig = plt.figure()
    ax = Axes3Ds(fig)
    world.draw(ax)

    # Draw a trajectory through the world.
    a = np.array([0.2, 0.3, 0.4])
    b = np.array([4.1, 6.1, 3.6])
    points = np.linspace(a, b, 100)
    ax.plot3D(points[:, 0], points[:, 1], points[:, 2], 'b.')

    # Illustrate any places where the trajectory gets close to an obstacle.
    (p, d) = world.closest_points(points)
    mask = d <= 0.5
    a = points[mask, :]
    b = p[mask, :]
    for i in range(a.shape[0]):
        ax.plot3D([a[i, 0], b[i, 0]], [a[i, 1], b[i, 1]], [a[i, 2], b[i, 2]], 'k')

    # Draw all plots.
    plt.show()
