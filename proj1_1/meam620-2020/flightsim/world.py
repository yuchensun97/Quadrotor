import json
import numpy as np

from flightsim.shapes import Cuboid

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
            return cls(json.load(file))

    def to_file(self, filename):
        """
        Write world definition to a .json text file.

        Parameters:
            filename

        Example use:
            my_word.to_file('my_filename.json')
        """
        with open(filename, 'w') as file:
            # Dump dict into stream for .json file.
            stream = json.dumps(self.world)
            # Adjust whitespace for more readable output (optional).
            stream = stream.replace('{"extents"','\n{"extents"')
            stream = stream.replace('"blocks"', '\n"blocks"')
            stream = stream.replace('"bounds"', '\n"bounds"')
            stream = stream.replace('}]}', '}]\n}')
            # Write file.
            file.write(stream)

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

        # This doesn't look nice because the z-order isn't sophisticated enough.
        # wireframe = Cuboid(ax, xmax-xmin, ymax-ymin, zmax-zmin, alpha=0, linewidth=1, edgecolors='k')
        # wireframe.transform((xmin,ymin,zmin))

        blocks = self.world['blocks']
        block_artists = []
        for b in blocks:
            (xmin, xmax, ymin, ymax, zmin, zmax) = b['extents']
            c = Cuboid(ax, xmax-xmin, ymax-ymin, zmax-zmin, alpha=0.6, linewidth=1, edgecolors='k')
            c.transform(position=(xmin, ymin, zmin))
            block_artists.extend(c.artists)
        return block_artists

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
        world_data = {'bounds':bounds, 'blocks':blocks}
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
            my_world = World.grid_forest()
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

        world_data = {'bounds':bounds, 'blocks':blocks}
        return cls(world_data)

if __name__ == '__main__':
    from axes3ds import Axes3Ds
    import matplotlib.pyplot as plt

    # Test grid_forest world.
    world = World.grid_forest(n_rows=4, n_cols=3, width=0.5, height=3.0, spacing=2.0)
    # Save to file.
    world.to_file('worlds/grid_forest.json')
    # Build a new world from the saved file.
    world = World.from_file('worlds/grid_forest.json')
    # Draw.
    fig = plt.figure()
    ax = Axes3Ds(fig)
    world.draw(ax)

    # Draw all plots.
    plt.show()
