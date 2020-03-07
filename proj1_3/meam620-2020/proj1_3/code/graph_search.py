from heapq import heappush, heappop  # Recommended.
import numpy as np

from math import sqrt

from flightsim.world import World
from proj1_3.code.occupancy_map import OccupancyMap # Recommended.

def graph_search(world, resolution, margin, start, goal, astar):
    """
    Parameters:
        world,      World object representing the environment obstacles
        resolution, xyz resolution in meters for an occupancy map, shape=(3,)
        margin,     minimum allowed distance in meters from path to obstacles.
        start,      xyz position in meters, shape=(3,)
        goal,       xyz position in meters, shape=(3,)
        astar,      if True use A*, else use Dijkstra
    Output:
        path,       xyz position coordinates along the path in meters with
                    shape=(N,3). These are typically the centers of visited
                    voxels of an occupancy map. The first point must be the
                    start and the last point must be the goal. If no path
                    exists, return None.
    """

    # While not required, we have provided an occupancy map you may use or modify.
    occ_map = OccupancyMap(world, resolution, margin)
    # Retrieve the index in the occupancy grid matrix corresponding to a position in space.
    start_index = tuple(occ_map.metric_to_index(start))
    start_cost = 0
    start_x,start_y,start_z = start_index
    goal_index = tuple(occ_map.metric_to_index(goal))
    goal_x,goal_y,goal_z = goal_index
    goal_cost = np.inf
    heuristic = sqrt((start_x-goal_x)**2+(start_y-goal_y)**2+(start_z-goal_z)**2)
    start_f = heuristic
    goal_f = np.inf
    # Initialization
    Q = []    # priority queue of open nodes
    G = []    # all the graph
    path = np.array([list(goal)]) # initialize path

    # world representing in voxel
    x = occ_map.map.shape[0]    
    y = occ_map.map.shape[1]
    z = occ_map.map.shape[2]    
    index = 0
    pos = np.zeros((x,y,z))    # position in a list

    class grid(object):
        """
        This class used to store features of a voxel
        """
        def __init__(self,idx,cost_to_come=np.inf,parent=np.NaN,heu=heuristic,cost=np.inf):
            self.idx = idx
            self.cost_to_come = cost_to_come
            self.parent = parent
            self.heu = heu
            self.cost = cost

    # store the features of all node in an object list
    for i in range(x):
        for j in range(y):
            for k in range(z):

                pos[i][j][k]=index    # store index

                if start_index == (i,j,k):
                    vox = grid(idx=(i,j,k),cost_to_come=0)
                    start_pos = index
                elif goal_index == (i,j,k):
                    vox = grid(idx=(i,j,k))
                    goal_pos = index
                else:
                    vox = grid(idx=(i,j,k))

                index+=1
                G.append(vox)    # store all the voxel object in a list *** Do Not Change the idx***
    
    if astar:
        heappush(Q,(start_f,G[start_pos].idx))
        heappush(Q,(goal_f,G[goal_pos].idx))

        # A* algorithm
        while (goal_cost,goal_index) in Q and Q[0][0]<np.inf:
            u = Q[0]    # node that has smllest cost
            ux,uy,uz = u[1] # node's coordination
            heappop(Q)    # alwasy pop out node that has smallest cost

            if (ux,uy,uz) == goal_index:
                break

            # push neigbor(u) into Q
            for dx in range(-1,2):
                for dy in range(-1,2):
                    for dz in range(-1,2):

                        if dx==0 and dy == 0 and dz==0:
                            continue
                        
                        # update new node
                        new_x = dx+ux
                        new_y = dy+uy
                        new_z = dz+uz
                        is_valid_neigbor = occ_map.is_valid_index((new_x,new_y,new_z))
                        if is_valid_neigbor:
                            is_occupy = occ_map.is_occupied_index((new_x,new_y,new_z))
                            if not is_occupy:
                                idx_in_G = int(pos[new_x][new_y][new_z])
                                v = G[idx_in_G]
                                c_u_v = sqrt(dx**2+dy**2+dz**2)
                                idx_of_u = int(pos[ux][uy][uz])
                                d = G[idx_of_u].cost_to_come+c_u_v
                                if d<v.cost_to_come:
                                    heuristic = sqrt((new_x-goal_x)**2+(new_y-goal_y)**2+(new_z-goal_z)**2)
                                    f = heuristic+d    # calculate the cost of a node
                                    G[idx_in_G] = grid((new_x,new_y,new_z),d,(ux,uy,uz),heuristic,f)    # update new node's feature
                                    heappush(Q,(G[idx_in_G].cost,G[idx_in_G].idx))    # update the open list

    else:
        heappush(Q,(start_cost,G[start_pos].idx))
        heappush(Q,(goal_cost,G[goal_pos].idx))

        # Dijstra Algorithms starts here#

        while (goal_cost,goal_index) in Q and Q[0][0]<np.inf:
            u = Q[0]    # node that has smllest cost
            ux,uy,uz = u[1] # node's coordination
            heappop(Q)    # alwasy pop out node that has smallest cost

            # push neigbor(u) into h
            for dx in range(-1,2):
                for dy in range(-1,2):
                    for dz in range(-1,2):

                        if dx==0 and dy == 0 and dz==0:
                            continue
                        
                        # update new node
                        new_x = dx+ux
                        new_y = dy+uy
                        new_z = dz+uz
                        is_valid_neigbor = occ_map.is_valid_index((new_x,new_y,new_z))
                        if is_valid_neigbor:
                            is_occupy = occ_map.is_occupied_index((new_x,new_y,new_z))
                            if not is_occupy:
                                idx_in_G = int(pos[new_x][new_y][new_z])
                                v = G[idx_in_G]
                                c_u_v = sqrt(dx**2+dy**2+dz**2)
                                idx_of_u = int(pos[ux][uy][uz])
                                d = G[idx_of_u].cost_to_come+c_u_v
                                if d<v.cost_to_come:
                                    G[idx_in_G] = grid((new_x,new_y,new_z),d,(ux,uy,uz))    # update new node's feature
                                    heappush(Q,(G[idx_in_G].cost_to_come,G[idx_in_G].idx))    # update open list
        
    # trace parent
    if np.any(np.isnan(G[goal_pos].parent)):
        return None
    else:
        parent_x,parent_y,parent_z = G[goal_pos].parent
        parent = occ_map.index_to_metric_center((parent_x,parent_y,parent_z))
        path = np.r_[path,np.array([list(parent)])]
        idx_in_G = int(pos[parent_x][parent_y][parent_z])
        while 1:
            parent_x,parent_y,parent_z = G[idx_in_G].parent
            parent = occ_map.index_to_metric_center((parent_x,parent_y,parent_z))
            path = np.r_[path,np.array([list(parent)])]
            idx_in_G = int(pos[parent_x][parent_y][parent_z])
            if idx_in_G == start_pos:
                break
        path = np.r_[path,np.array([list(start)])]
        path = np.flipud(path)
        
        return path