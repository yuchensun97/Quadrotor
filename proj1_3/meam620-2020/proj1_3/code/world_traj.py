import numpy as np

from proj1_3.code.occupancy_map import OccupancyMap # Recommended.
from proj1_3.code.graph_search import graph_search
from math import ceil

class WorldTraj(object):
    """

    """
    def __init__(self, world, start, goal):
        """
        This is the constructor for the trajectory object. A fresh trajectory
        object will be constructed before each mission. For a world trajectory,
        the input arguments are start and end positions and a world object. You
        are free to choose the path taken in any way you like.

        You should initialize parameters and pre-compute values such as
        polynomial coefficients here.

        Parameters:
            world, World object representing the environment obstacles
            start, xyz position in meters, shape=(3,)
            goal,  xyz position in meters, shape=(3,)

        """

        # You must choose resolution and margin parameters to use for path
        # planning. In the previous project these were provided to you; now you
        # must chose them for yourself. Your may try these default values, but
        # you should experiment with them!
        self.resolution = np.array([0.2, 0.2, 0.2])
        self.margin = 0.4

        # You must store the dense path returned from your Dijkstra or AStar
        # graph search algorithm as an object member. You will need it for
        # debugging, it will be used when plotting results.
        self.path = graph_search(world, self.resolution, self.margin, start, goal, astar=True)

        # You must generate a sparse set of waypoints to fly between. Your
        # original Dijkstra or AStar path probably has too many points that are
        # too close together. Store these waypoints as a class member; you will
        # need it for debugging and it will be used when plotting results.
        self.points = np.zeros((1,3)) # shape=(n_pts,3)
        self.points = self.path

        # Finally, you must compute a trajectory through the waypoints similar
        # to your task in the first project. One possibility is to use the
        # WaypointTraj object you already wrote in the first project. However,
        # you probably need to improve it using techniques we have learned this
        # semester.

        # STUDENT CODE HERE
        occ_map =  OccupancyMap(world, self.resolution, self.margin)

        def check_collision(p1,p2):
            """
            This function check whether the connection of two waypoints collides with each other
            Args:
                p1: coordinates of point 1, a shape(3,) array
                p2: coordinates of point 2, a shape(3,) array
            Returns:
                1: The connection of two waypoint collide
                0: The connection of two waypoint does not collide
            """
            seg_length = 0.1
            distance = np.linalg.norm(p2-p1)
            unit_vec = (p2-p1)/distance
            segment = ceil(distance/seg_length)
            seg_length = distance/segment    # re-define the length of each segment

            # store segment points
            for i in range(1,segment):
                seg_point = p1+i*seg_length*unit_vec
                seg_point_x = seg_point[0]
                seg_point_y = seg_point[1]
                seg_point_z = seg_point[2]
                seg_point = (seg_point_x,seg_point_y,seg_point_z)
                is_valid = occ_map.is_valid_metric(seg_point)
                is_occupied = occ_map.is_occupied_metric(seg_point)
                if is_valid and not (is_occupied):
                    bool_collide = 0
                else:
                    bool_collide = 1
                    break
            
            return bool_collide

        # optimize path
        check_point = start
        check_point_idx = 0
        check_goal = goal
        idx = len(self.path)-1
        goal_idx = idx
        new_path = np.array([start])
        while not (check_point==goal).all():
            while check_collision(check_point,check_goal) and idx-check_point_idx>1:
                idx = idx -1
                check_goal = self.path[idx]
            check_point = check_goal
            check_point_idx = idx
            check_goal = goal
            idx = goal_idx
            new_path = np.r_[new_path,[check_point]]

        self.path = new_path

        # quint trajectory starts here
        self.pt = self.path # type: np.array
        self.num_pts = len(self.pt)
        self.acc_mean = 3.8 # the mean acceleartion 
        self.t_segment = [0]
        self.t_between_pt = []
        time_diff = 0
        for i in range(0,self.num_pts-1):
            time_diff = 2*np.sqrt(np.linalg.norm(self.pt[i+1]-self.pt[i]))/self.acc_mean
            self.t_segment.append(time_diff+self.t_segment[i]) # time for waypoint to reach
            self.t_between_pt.append(time_diff) # time different between two points

        tg = self.t_between_pt[-1]

        A = np.zeros((6*(self.num_pts-1),6*(self.num_pts-1)))

        # boundary constrain
        A[0,5]=1
        A[1,4]=1
        A[2,3]=2

        A[-3,-6:] = np.array([tg**5,tg**4,tg**3,tg**2,tg,1])
        A[-2,-6:] = np.array([5*tg**4,4*tg**3,3*tg**2,2*tg,1,0])
        A[-1,-6:] = np.array([20*tg**3,12*tg**2,6*tg,2,0,0])

        # continuous constrain
        for i in range(self.num_pts-2):
            tg = self.t_between_pt[i]

            # position constrain
            A[2*i+3,i*6:(i+1)*6]=np.array([tg**5,tg**4,tg**3,tg**2,tg,1])
            A[2*i+4,(i+2)*6-1] = 1

            # velosity constrain
            A[3+(self.num_pts-2)*2+i,i*6:(i+1)*6]=np.array([5*tg**4,4*tg**3,3*tg**2,2*tg,1,0])
            A[3+(self.num_pts-2)*2+i,(i+2)*6-2] = -1

            # acceleration constrain
            A[3+(self.num_pts-2)*3+i,i*6:(i+1)*6] = np.array([20*tg**3,12*tg**2,6*tg,2,0,0])
            A[3+(self.num_pts-2)*3+i,(i+2)*6-3] = -2

            # jerk constrain
            A[3+(self.num_pts-2)*4+i,i*6:(i+1)*6] = np.array([60*tg**2,24*tg,6,0,0,0])
            A[3+(self.num_pts-2)*4+i,(i+2)*6-4] = -6

            # snap constrain
            A[3+(self.num_pts-2)*5+i,i*6:(i+1)*6] = np.array([120*tg,24,0,0,0,0])
            A[3+(self.num_pts-2)*5+i,(i+2)*6-5] = -24

        # P vector
        px = np.zeros((6*(self.num_pts-1),1))
        py = np.zeros((6*(self.num_pts-1),1))
        pz = np.zeros((6*(self.num_pts-1),1))

        px[0,0]=start[0]
        py[0,0]=start[1]
        pz[0,0]=start[2]

        px[-3,0]=goal[0]
        py[-3,0]=goal[1]
        pz[-3,0]=goal[2]

        for i in range(self.num_pts-2):
            px[2*i+3,0] = new_path[i+1,0]
            px[2*i+4,0] = new_path[i+1,0]

            py[2*i+3,0] = new_path[i+1,1]
            py[2*i+4,0] = new_path[i+1,1]

            pz[2*i+3,0] = new_path[i+1,2]
            pz[2*i+4,0] = new_path[i+1,2]

        self.Cx = np.linalg.inv(A)@px
        self.Cy = np.linalg.inv(A)@py
        self.Cz = np.linalg.inv(A)@pz


    def update(self, t):
        """
        Given the present time, return the desired flat output and derivatives.

        Inputs
            t, time, s
        Outputs
            flat_output, a dict describing the present desired flat outputs with keys
                x,        position, m
                x_dot,    velocity, m/s
                x_ddot,   acceleration, m/s**2
                x_dddot,  jerk, m/s**3
                x_ddddot, snap, m/s**4
                yaw,      yaw angle, rad
                yaw_dot,  yaw rate, rad/s
        """
        x        = np.zeros((3,))
        x_dot    = np.zeros((3,))
        x_ddot   = np.zeros((3,))
        x_dddot  = np.zeros((3,))    # jerk
        x_ddddot = np.zeros((3,))    # snap
        yaw = 0
        yaw_dot = 0

        # STUDENT CODE HERE

        pos = self.pt # type: np.array
        for i in range(0,self.num_pts-1):
            if t >= self.t_segment[-1]:
                x = pos[-1]
                x_dot = np.zeros((3,))
                x_ddot = np.zeros((3,))
                x_dddot = np.zeros((3,))
                x_ddddot = np.zeros((3,))

            elif t>=self.t_segment[i] and t<self.t_segment[i+1]:
                t_now = t-self.t_segment[i]
                Cx = self.Cx[i*6:(i+1)*6,0]
                Cy = self.Cy[i*6:(i+1)*6,0]
                Cz = self.Cz[i*6:(i+1)*6,0]

                position = np.array([[t_now**5,t_now**4,t_now**3,t_now**2,t_now,1]])
                velocity = np.array([[5*t_now**4,4*t_now**3,3*t_now**2,2*t_now,1,0]])
                acceleration = np.array([[20*t_now**3,12*t_now**2,6*t_now,2,0,0]])
                jerk = np.array([[60*t_now**2,24*t_now,6,0,0,0]])
                snap = np.array([[120*t_now,24,0,0,0,0]])
                
                x1 = position@Cx
                y = position@Cy
                z = position@Cz
                x = np.array([x1,y,z]).flatten()

                x1d = velocity@Cx
                yd = velocity@Cy
                zd = velocity@Cz
                x_dot = np.array([x1d,yd,zd]).flatten()

                x1dd = acceleration@Cx
                ydd = acceleration@Cy
                zdd = acceleration@Cz
                x_ddot = np.array([x1dd,ydd,zdd]).flatten()

                x1ddd = jerk@Cx
                yddd = jerk@Cy
                zddd = jerk@Cz
                x_dddot = np.array([x1ddd,yddd,zddd]).flatten()

                x1dddd = snap@Cx
                ydddd = snap@Cy
                zdddd = snap@Cz
                x_ddddot = np.array([x1dddd,ydddd,zdddd]).flatten()

        flat_output = { 'x':x, 'x_dot':x_dot, 'x_ddot':x_ddot, 'x_dddot':x_dddot, 'x_ddddot':x_ddddot,
                        'yaw':yaw, 'yaw_dot':yaw_dot}
        return flat_output
