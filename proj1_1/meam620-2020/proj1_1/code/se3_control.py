import numpy as np
from scipy.spatial.transform import Rotation

class SE3Control(object):
    """

    """
    def __init__(self, quad_params):
        """
        This is the constructor for the SE3Control object. You may instead
        initialize any parameters, control gain values, or private state here.

        For grading purposes the controller is always initialized with one input
        argument: the quadrotor's physical parameters. If you add any additional
        input arguments for testing purposes, you must provide good default
        values!

        Parameters:
            quad_params, dict with keys specified by crazyflie_params.py

        """

        # Quadrotor physical parameters.
        self.mass            = quad_params['mass'] # kg
        self.Ixx             = quad_params['Ixx']  # kg*m^2
        self.Iyy             = quad_params['Iyy']  # kg*m^2
        self.Izz             = quad_params['Izz']  # kg*m^2
        self.arm_length      = quad_params['arm_length'] # meters
        self.rotor_speed_min = quad_params['rotor_speed_min'] # rad/s
        self.rotor_speed_max = quad_params['rotor_speed_max'] # rad/s
        self.k_thrust        = quad_params['k_thrust'] # N/(rad/s)**2
        self.k_drag          = quad_params['k_drag']   # Nm/(rad/s)**2

        # You may define any additional constants you like including control gains.
        self.inertia = np.diag(np.array([self.Ixx, self.Iyy, self.Izz])) # kg*m^2
        self.g = 9.81 # m/s^2

        # STUDENT CODE HERE
        self.gamma = self.k_drag/self.k_thrust
        self.Kp = np.diag(np.array([6.0,6.0,7.3]))
        self.Kd = np.diag(np.array([3.9,3.9,5.6]))
        self.Kr = np.diag(np.array([230,230,50]))
        self.Kw = np.diag(np.array([20,20,25]))
        self.u2Matrix = np.array([[1,1,1,1],[0,self.arm_length,0,-self.arm_length],[-self.arm_length,0,self.arm_length,0],[self.gamma,-self.gamma,self.gamma,-self.gamma]])

    def update(self, t, state, flat_output):
        """
        This function receives the current time, true state, and desired flat
        outputs. It returns the command inputs.

        Inputs:
            t, present time in seconds
            state, a dict describing the present state with keys
                x, position, m
                v, linear velocity, m/s
                q, quaternion [i,j,k,w]
                w, angular velocity, rad/s
            flat_output, a dict describing the present desired flat outputs with keys
                x,        position, m
                x_dot,    velocity, m/s
                x_ddot,   acceleration, m/s**2
                x_dddot,  jerk, m/s**3
                x_ddddot, snap, m/s**4
                yaw,      yaw angle, rad
                yaw_dot,  yaw rate, rad/s

        Outputs:
            control_input, a dict describing the present computed control inputs with keys
                cmd_motor_speeds, rad/s
                cmd_thrust, N (for debugging and laboratory; not used by simulator)
                cmd_moment, N*m (for debugging; not used by simulator)
                cmd_q, quaternion [i,j,k,w] (for laboratory; not used by simulator)
        """
        cmd_motor_speeds = np.zeros((4,))
        cmd_thrust = 0
        cmd_moment = np.zeros((3,))
        cmd_q = np.zeros((4,))

        # STUDENT CODE HERE

        # extract the value of state
        x_state = state['x']
        v = state['v']
        q = state['q']
        w = state['w']

        # extract the value of desired output
        x_flat = flat_output['x']
        x_dot = flat_output['x_dot']
        x_ddot = flat_output['x_ddot']
        x_dddot = flat_output['x_dddot']
        x_ddddot = flat_output['x_ddddot']
        yaw = flat_output['yaw']
        yaw_dot = flat_output['yaw_dot']

        # computing F des
        # r trajectory
        r_T = x_flat.reshape(x_flat.shape[0],1)
        r_dT = x_dot.reshape(x_dot.shape[0],1)
        r_ddT = x_ddot.reshape(x_ddot.shape[0],1)
        # state estimate r
        r = x_state.reshape(x_state.shape[0],1)
        r_d = v.reshape(v.shape[0],1)
        # calculate rdes_ddot
        rdes_ddot = r_ddT-self.Kd@(r_d-r_dT)-self.Kp@(r-r_T)
        Fdes = self.mass*rdes_ddot+np.array([[0],[0],[self.mass*self.g]]) # column

        # compute u1
        R_matrix = Rotation.from_quat(q)
        R_matrix = R_matrix.as_dcm()
        b3 = R_matrix@np.array([[0],[0],[1]])
        b3_T = np.transpose(b3)
        u1 = b3_T@Fdes # a number

        # determine Rdes
        b3_des = Fdes/np.linalg.norm(Fdes) # column
        b3_des_T = np.transpose(b3_des) # row, for computing
        cos_yaw = np.cos(np.array([yaw]))
        cos_yaw = cos_yaw[0]
        sin_yaw = np.sin(np.array([yaw]))
        sin_yaw = sin_yaw[0]
        a_yaw = np.array([[cos_yaw,sin_yaw,0]]) # row
        b2_des_T = np.cross(b3_des_T,a_yaw)/np.linalg.norm(np.cross(b3_des_T,a_yaw)) # row
        b1_des_T = np.cross(b2_des_T,b3_des_T)
        R_des_T = np.array([b1_des_T[0],b2_des_T[0],b3_des_T[0]]) # transpose of R
        R_des = np.transpose(R_des_T)

        # find eR
        eR_vee = 1/2*(R_des_T@R_matrix-R_matrix.T@R_des)
        eR = np.array([[eR_vee[2,1]],[eR_vee[0,2]],[eR_vee[1,0]]]) # column

        # find u2
        omega = w.reshape(w.shape[0],1)
        u2 = self.inertia@(-self.Kr@eR-self.Kw@omega) # column
        u = np.array([[u1[0]],[u2[0,0]],[u2[1,0]],[u2[2,0]]])

        # compute F1, F2, F3 and F4
        F = np.linalg.inv(self.u2Matrix)@u
        cmd_thrust = np.array([F[0,0],F[1,0],F[2,0],F[3,0]])

        cmd_motor_speeds=[]
        # compute motor speed:
        for i in range(0,4):
            if cmd_thrust[i,0]>0:
                cmd_motor_speeds.append(np.sqrt(cmd_thrust[i,0]/self.k_thrust))
            else:
                cmd_motor_speeds.append(0)

                
        cmd_motor_speeds = np.array(cmd_motor_speeds)
        cmd_thrust = F[0,0]+F[1,0]+F[2,0]+F[3,0]

        # cmd_motor_speeds=np.array([0,0,0,0])
        # compute motor moment
        cmd_moment = np.array([u2[0,0],u2[1,0],u2[2,0]])

        # compute motor orientation
        cmd_q = Rotation.from_dcm(R_des)
        cmd_q = cmd_q.as_quat()

        control_input = {'cmd_motor_speeds':cmd_motor_speeds,
                         'cmd_thrust':cmd_thrust,
                         'cmd_moment':cmd_moment,
                         'cmd_q':cmd_q}
        return control_input