# MEAM620-Advanced-Robotics
This project is about quadrotor. Phase 1 contains detailed information about non-linear controller and flatteness, Dijstra and A* implementation in 3D environment with obstacles, minimum jerk trajectory.  

**What's next?**  
Perception, mapping and SLAM  

To run this project on your local end, please make sure you have install the following package:  
* numpy
* cvxopt
* matplotlib
* scipy
* timeout_decorator
## Phase 1_1
* **se3_control.py**: non-linear controller with flat output. Author@ Yuchen Sun
* **waypoint_traj.py**: constant velocity trajecory. Author@ Yuchen Sun
* **sandbox.py**: simulator. Author@ MEAM620 Staff
## Phase 1_2
* **graph_search**: Implementation of Dijstra and A*. Author@Yuchen Sun
* **occupancy_map**: Provide the enviroment for path planning problem. Author@ MEAM620 Staff
* **sandbox.py**: simulator. Author@ MEAM620 Staff
## Phase 1_3
* **graph_search.py**: same as Phase 1_2
* **se3_control.py**: same as Phase 1_1, except I adjust some parameter of PID.
* **waypoint_traj.py**: optimize path return from A* and implement minimum jerk trajectory. Author@ Yuchen Sun
* **sandbox.py**: simulator. Author@ Meam 620 Staff
