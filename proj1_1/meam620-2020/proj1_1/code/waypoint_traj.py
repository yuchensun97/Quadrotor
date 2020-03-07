import numpy as np

class WaypointTraj(object):
    """

    """
    def __init__(self, points):
        """
        This is the constructor for the Trajectory object. A fresh trajectory
        object will be constructed before each mission. For a waypoint
        trajectory, the input argument is an array of 3D destination
        coordinates. You are free to choose the times of arrival and the path
        taken between the points in any way you like.

        You should initialize parameters and pre-compute values such as
        polynomial coefficients here.

        Inputs:
            points, (N, 3) array of N waypoint coordinates in 3D
        """

        # STUDENT CODE HERE
        self.pt = points # type: np.array
        self.num_pts = len(points)
        self.velocity = 1.5 # the norm of the velocity
        self.t_segment = [0]
        self.t_between_pt = []
        time_diff = 0
        for i in range(0,self.num_pts-1):
            time_diff = np.linalg.norm(self.pt[i+1]-self.pt[i])/self.velocity
            self.t_segment.append(time_diff+self.t_segment[i]) # time for waypoint to reach
            self.t_between_pt.append(time_diff) # time different between two points



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
        x_dddot  = np.zeros((3,))
        x_ddddot = np.zeros((3,))
        yaw = 0
        yaw_dot = 0


        # STUDENT CODE HERE
        pos = self.pt # type: np.array
        for i in range(0,self.num_pts-1):
            if t >= self.t_segment[-1]:
                x = pos[-1]
                x_dot = np.zeros((3,))
            elif t>=self.t_segment[i] and t<self.t_segment[i+1]:
                x_dot = (pos[i+1]-pos[i])/self.t_between_pt[i]
                x = x_dot*(t-self.t_segment[i])+pos[i]
        
        print(x_dot)
        
        flat_output = { 'x':x, 'x_dot':x_dot, 'x_ddot':x_ddot, 'x_dddot':x_dddot, 'x_ddddot':x_ddddot,
                        'yaw':yaw, 'yaw_dot':yaw_dot}
        return flat_output
