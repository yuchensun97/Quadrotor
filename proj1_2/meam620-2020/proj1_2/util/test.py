import contextlib
import inspect
import json
import os
from pathlib import Path
import time
import unittest

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from scipy.spatial.distance import cdist

from flightsim.world import World
from proj1_2.code.occupancy_map import OccupancyMap
from proj1_2.code.graph_search import graph_search as student_graph_search_fn

def set_path_metrics(metrics, path_name, path, time, world, start, goal, resolution, margin, expected_path_length):
    """
    Set path metrics in one easy function!
    :return:
        metrics filled for the given path.
    """

    metrics[path_name] = {}
    metrics[path_name]['time'] = time

    eps = 1e-3

    if path is not None:
        metrics[path_name]['path_length'] = float(round(np.sum(np.linalg.norm(np.diff(path, axis=0),axis=1)),3))
        metrics[path_name]['reached_start'] = bool(np.linalg.norm(path[0] - start) <= 1e-3)
        metrics[path_name]['reached_goal'] = bool(np.linalg.norm(path[-1] - goal) <= 1e-3)
        metrics[path_name]['no_collision'] = not np.any(world.path_collisions(path, margin)[0])
    else:
        metrics[path_name]['path_length'] = np.inf
        metrics[path_name]['reached_start'] = False
        metrics[path_name]['reached_goal'] = False
        metrics[path_name]['no_collision'] = False

    if expected_path_length is not None:
        if expected_path_length == np.inf and metrics[path_name]['path_length'] == np.inf:
            metrics[path_name]['is_optimal'] = True
        else:
            metrics[path_name]['is_optimal'] = bool(metrics[path_name]['path_length'] <= expected_path_length + max(
            resolution))
    else:
        metrics[path_name]['is_optimal'] = None  # Solution length not available for student-written tests.
    return metrics


def test_mission(graph_search_fn, world, start, goal, resolution, margin, expected_path_length):
    """
    Test the provided graph_search function against a world, start, and goal.
    Return the simulation results and the performance metrics.
    """

    # Run student code.
    oc = OccupancyMap(world, resolution, margin)
    results = {}

    start_time = time.time()
    results['dijkstra_path'] = graph_search_fn(world, resolution, margin, start, goal, False)
    dijkstra_time = round(time.time() - start_time, 3)

    start_time = time.time()
    results['astar_path'] = graph_search_fn(world, resolution, margin, start, goal, True)
    astar_time = round(time.time() - start_time, 2)

    metrics = {}
    # Evaluate results for Dijkstra and Astar
    set_path_metrics(metrics, 'dijkstra', results['dijkstra_path'], dijkstra_time, world, start, goal, resolution, margin, expected_path_length)
    set_path_metrics(metrics, 'astar', results['astar_path'], astar_time, world, start, goal, resolution, margin, expected_path_length)

    w = world.world['bounds']['extents']
    metrics['map_nodes'] = int(np.prod(np.round((w[1::2]-w[0::2])/resolution)))

    return results, metrics


def plot_mission(world, start, goal, results, test_name):
    """
    Return a figure showing path through trees along with start and end.
    """

    import matplotlib.pyplot as plt
    from matplotlib.axes import Axes
    from matplotlib.projections import register_projection

    from flightsim.axes3ds import Axes3Ds

    def plot_path(path, path_type):
        fig = plt.figure()
        ax = Axes3Ds(fig)
        world.draw(ax)
        if path is not None:
            world.draw_line(ax, path, color='blue')
            world.draw_points(ax, path, color='blue')
        ax.plot([start[0]], [start[1]], [start[2]], 'go', markersize=10, markeredgewidth=3, markerfacecolor='none')
        ax.plot( [goal[0]],  [goal[1]],  [goal[2]], 'ro', markersize=10, markeredgewidth=3, markerfacecolor='none')
        ax.set_title("%s Path through %s" % (path_type, test_name))
        return fig

    dijkstra_path = results['dijkstra_path']
    astar_path = results['astar_path']
    dijkstra_fig = plot_path(dijkstra_path, "Dijkstra's")
    astar_fig = plot_path(astar_path, "A*")
    return [dijkstra_fig, astar_fig]


class TestBase(unittest.TestCase):
    graph_search_fn = staticmethod(student_graph_search_fn)  # Keep the function unbound.

    longMessage = False
    outpath = Path(inspect.getsourcefile(test_mission)).resolve().parent.parent / 'data_out'
    outpath.mkdir(parents=True, exist_ok=True)

    test_names = []

    def helper_test(self, test_name, world, start, goal, resolution, margin, expected_path_length):
        """
        Test student's graph_search against given world, start, and goal.
        Run solution, save metrics to file, save result plots to file.
        """
        with contextlib.redirect_stdout(None):  # Context gobbles stdout.
            (results, metrics) = test_mission(self.graph_search_fn, world, start, goal, resolution, margin,
                                              expected_path_length)
            with open(self.outpath / ('result_' + test_name + '.json'), 'w') as f:
                json.dump(metrics, f, indent=4, separators=(',', ': '))
            figs = plot_mission(world, start, goal, results, test_name)
            # Save all figures to file
            with PdfPages(self.outpath / ('result_' + test_name + '.pdf')) as pdf:
                for fig in figs:
                    pdf.savefig(fig)

    @classmethod
    def load_tests(cls, files):
        """
        Add one test for each input file. For each input file named
        "test_XXX.json" creates a new test member function that will generate
        output files "result_XXX.json" and "result_XXX.pdf".
        """
        for file in files:
            if file.stem.startswith('test_'):
                test_name = file.stem[5:]
                cls.test_names.append(test_name)
                world=World.from_file(file)

                # Dynamically add member function for this test.
                def fn(self, test_name=test_name,
                       world=world,
                       start=world.world['start'],
                       goal=world.world['goal'],
                       resolution=world.world['resolution'],
                       margin=world.world['margin'],
                       expected_path_length=world.world.get('expected_path_length', None)):
                    self.helper_test(test_name, world, start, goal, resolution, margin, expected_path_length)

                setattr(cls, 'test_' + test_name, fn)
                # Remove any pre-existing output files for this test.
                # TODO: The 'missing_ok' argument becomes available in Python
                # 3.8, at which time contextlib is no longer needed.
                with contextlib.suppress(FileNotFoundError):
                    (cls.outpath / ('result_' + test_name + '.json')).unlink()
                with contextlib.suppress(FileNotFoundError):
                    (cls.outpath / ('result_' + test_name + '.pdf')).unlink()

    @classmethod
    def collect_results(cls):
        results = []
        for name in cls.test_names:
            p = cls.outpath / ('result_' + name + '.json')
            if p.exists():
                with open(p) as f:
                    data = json.load(f)
                    data['test_name'] = name
                    results.append(data)
            else:
                results.append({'test_name': name})
        return results

    @classmethod
    def print_results(cls):
        results = cls.collect_results()
        for r in results:
            print()  # prettiness
            if len(r.keys()) > 1:
                if r['test_name'] != 'impossible':
                    dijkstra_optimal = r['dijkstra']['is_optimal'] or r['dijkstra']['is_optimal'] == None
                    astar_optimal = r['astar']['is_optimal'] or r['astar']['is_optimal'] == None
                    passed = r['dijkstra']['reached_start'] and r['dijkstra']['reached_goal'] and \
                             r['dijkstra']['no_collision'] and dijkstra_optimal and \
                             r['astar']['reached_start'] and r['astar']['reached_goal'] and \
                             r['astar']['no_collision'] and astar_optimal
                    print('{} {}, (size {:,})'.format('pass' if passed else 'FAIL', r['test_name'], r['map_nodes']))
                    for name in ['dijkstra', 'astar']:
                        print('    {} reached start: {}'.format(name, 'pass' if r[name]['reached_start'] else 'FAIL'))
                        print('    {} reached goal:  {}'.format(name, 'pass' if r[name]['reached_goal'] else 'FAIL'))
                        print('    {} no collision:  {}'.format(name, 'pass' if r[name]['no_collision'] else 'FAIL'))
                        print('    {} is optimal:    {}'.format(name,
                            {True: 'pass', False: 'FAIL', None: '?'}[r[name]['is_optimal']]))
                        print('    {} path length:   {}'.format(name, r[name]['path_length']))
                        print('    {} time, seconds: {}'.format(name, r[name]['time']))
                else:
                    passed = r['dijkstra']['path_length'] == np.inf and r['astar']['path_length'] == np.inf
                    print('{} {}, (size {:,})'.format('pass' if passed else 'FAIL', r['test_name'], r['map_nodes']))
                    for name in ['dijkstra', 'astar']:
                        print('    {} is optimal:    {}'.format(name,
                            {True: 'pass', False: 'FAIL', None: '?'}[r[name]['is_optimal']]))
                        print('    {} path length:   {}'.format(name, r[name]['path_length']))
                        print('    {} time, seconds: {}'.format(name, r[name]['time']))
            else:
                print("FAIL {name}\n"
                      "    Test failed with no results. Review error log.".format(
                    name=r['test_name']))


if __name__ == '__main__':
    """
    Run a test for each "test_*.json" file in this directory. You can add new
    tests by copying and editing these files.
    """

    # Collect tests.
    testpath = Path(inspect.getsourcefile(TestBase)).parent.resolve()
    testfiles = Path(testpath).glob('test_*.json')
    TestBase.load_tests(testfiles)

    # Run tests, results saved in data_out.
    suite = unittest.TestLoader().loadTestsFromTestCase(TestBase)
    unittest.TextTestRunner(verbosity=2).run(suite)

    # Collect results for display.
    TestBase.print_results()
