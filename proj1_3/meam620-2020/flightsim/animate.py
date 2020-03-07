"""
TODO: Set up figure for appropriate target video size (eg. 720p).
TODO: Decide which additional user options should be available.
TODO: Implement progress bar to show animation is working.
"""

from datetime import datetime

import numpy as np
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

from flightsim.axes3ds import Axes3Ds
from flightsim.shapes import Quadrotor

def _decimate_index(time, sample_time):
    """
    Given sorted lists of source times and sample times, return indices of
    source time closest to each sample time.
    """
    index = np.arange(time.size)
    sample_index = np.round(np.interp(sample_time, time, index)).astype(int)
    return sample_index

def animate(time, position, rotation, world, filename=None, blit=True, show_axes=True):
    """
    Animate a completed simulation result based on the time, position, and
    rotation history. The animation may be viewed live or saved to a .mp4 video
    (slower, requires additional libraries).

    Parameters
        time, (N,) with uniform intervals
        position, (N,3)
        rotation, (N,3,3)
        world, a World object
        filename, for saved video, or live view if None
        blit, if True use blit for faster animation, default is True
        show_axes, if True plot axes, default is True
    """

    # Visual style.
    shade = True

    # Temporal style.
    rtf = 1.0 # real time factor > 1.0 is faster than real time playback
    render_fps = 30
    close_on_finish = False

    # Decimate data to render interval; always include t=0.
    if time[-1] != 0:
        sample_time = np.arange(0, time[-1], 1/render_fps * rtf)
    else:
        sample_time = np.zeros((1,))
    index = _decimate_index(time, sample_time)
    time = time[index]
    position = position[index,:]
    rotation = rotation[index,:,:]

    # Set up axes.
    if filename is not None:
        fig = plt.figure(filename)
    else:
        fig = plt.figure('Animation')
    ax = Axes3Ds(fig)
    if not show_axes:
        ax.set_axis_off()
    ax.set_xlim(-1,1)
    ax.set_ylim(-1,1)
    ax.set_zlim(-1,1)

    quad = Quadrotor(ax)

    world_artists = world.draw(ax)

    title_artist = ax.set_title('t = {}'.format(time[0]))

    def init():
        ax.draw(fig.canvas.get_renderer())
        return world_artists + list(quad.artists) + [title_artist]

    def update(frame):
        title_artist.set_text('t = {:.2f}'.format(time[frame]))
        quad.transform(position=position[frame,:], rotation=rotation[frame,:,:])
        [a.do_3d_projection(fig.canvas.get_renderer()) for a in quad.artists]
        if close_on_finish and frame == time.size-1:
            plt.close(fig)
        return world_artists + list(quad.artists) + [title_artist]

    ani = FuncAnimation(fig=fig,
                        func=update,
                        frames=time.size,
                        init_func=init,
                        interval=1000.0/render_fps,
                        repeat=False,
                        blit=blit)
    if filename is not None:
        print('Saving Animation')
        ani.save(filename,
                 writer='ffmpeg',
                 fps=render_fps,
                 dpi=100)

def test(world, filename):

    def dummy_sim_result():
        t = np.arange(0, 10, 1/100)
        position = np.zeros((t.shape[0], 3))
        position[:,0] = np.cos(t)
        position[:,1] = np.sin(t)
        theta = t % (2*np.pi)
        rotvec = theta * np.array([[0], [0], [1]])
        rotation = Rotation.from_rotvec(rotvec.T).as_dcm()
        return (t, position, rotation)

    (time, position, rotation) = dummy_sim_result()

    wall_start = datetime.now()
    animate(time, position, rotation, world, filename)
    wall_elapsed = (datetime.now() - wall_start).total_seconds()
    wall_fps = time.size / wall_elapsed
    print('render time = {:.0f} s'.format(wall_elapsed))
    print('render fps = {:.0f}'.format(wall_fps))

if __name__ == '__main__':
    from flightsim.world import World

    world = World.from_file('worlds/grid_forest.json')

    print('Test .mp4 rendering.')
    test(world=world, filename='data_out/quadrotor.mp4')
    plt.show()

    print('Test live view.')
    test(world=world, filename=None)
    plt.show()
