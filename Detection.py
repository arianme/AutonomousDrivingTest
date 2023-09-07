import numpy as np
import os
import cv2
import open3d as o3d
from Pose import Pose
from filterpy.common import Q_discrete_white_noise
import copy

class Detection:
    def __init__(self, det_class, pose, v = np.nan) -> None:
        self.det_class = det_class
        self.latest_pose = pose
        self.latest_v = v #(2,N)
        self.prev_poses = []
        self.prev_v = []
        self.lost_counter = 0
        #Kalman variables
        self.kalman = False #Has been initialized
        self.predict = False #Position is only predicted

    def update_pose(self, new_pose):
        self.lost_counter = 0
        self.latest_pose = new_pose

    def update_v(self, new_v):
        self.latest_v = new_v
    
    #Counter for lost tracking - Gets set to 0 everytime there is a pose update
    def next_frame(self):
        self.lost_counter += 1
        self.prev_poses.append(copy.copy(self.latest_pose))
        self.prev_v.append(copy.copy(self.latest_v))

        self.latest_pose.reset()
        self.latest_v = np.nan
        
    

    #Low pass filter for the velocity, replaces all velocities with the filtered mean
    def lp_filter_v(self):
        past = 5
        a = 0.6
        if not np.isnan(self.latest_v).any():
            #Mean Velocity
            #speed = np.linalg.norm(self.latest_v, axis=0)
            if self.latest_v.ndim > 1:
                v_mean = np.mean(self.latest_v, axis=1)
            for i in reversed(range(np.clip(len(self.prev_v)-past, a_min=0, a_max=None), len(self.prev_v))):
                #Check if there is any past v reading that is not nan
                if not np.isnan(self.prev_v[i]).any():
                    if self.prev_v[i].ndim > 1:
                        prev_v_read = np.mean(self.prev_v[i], axis=1)
                    else:
                        prev_v_read = self.prev_v[i]
                    lp_v = a*prev_v_read + (1-a)*v_mean
                    self.latest_v = lp_v
                    return
            #If no non-nan previous measurements found, average over current readings
            if self.latest_v.ndim > 1:
                self.latest_v = np.mean(self.latest_v, axis=1)
    
    #################################
    ################################# Kalman Filter
    #This requires at least one past measurement to set the velocity
    def kalman_init(self, param_dt=0.05):
        self.dt = param_dt
        hist = len(self.prev_poses)
        if hist < 1 or self.kalman:
            #print("Not enough tracking history.")
            return 
        #Ensure there is a tracking frame with 3D data
        if np.isnan(self.latest_pose.center_3d).any():
            #print("Latest pose is ", self.latest_pose.center_3d)
            return
        index = -1
        for i in reversed(range(len(self.prev_poses))):
            if not np.isnan(self.prev_poses[i].center_3d).any():
                index = i
                break
        if index == -1:
            #print("No history frame found with non nan 3D position.")
            return
        self.kalman = True
        #Use 1st order derivative of sensor measurements for velocity
        v_init = (self.latest_pose.center_3d[:2] - self.prev_poses[index].center_3d[:2]) / self.dt

        p_init = self.latest_pose.center_3d
        p_init = np.array([0, 0, 0])
        v_init = np.array([0, 0, 0])

        #No control transition matrix B, because we have no control/acceleration
        dt = self.dt

        #Initialize State Vector to 0 for (x, y, z, vx, vy, vz, ax, ay, az), z velocity is set to 0
        self.X = np.array([p_init[0], p_init[1], p_init[2], v_init[0], v_init[1], 0, 0, 0, 0])


        #State Covariance Matrix, initial uncertainty set to the range if LiDAR
        pv = 4000
        self.P = np.eye(9, 9)*pv

        #State Transition Matrix
        self.A = np.array([[1, 0, 0, dt , 0  , 0  , dt**2/2 , 0       , 0       ],
                           [0, 1, 0, 0  , dt , 0  , 0       , dt**2/2 , 0       ],
                           [0, 0, 1, 0  , 0  , dt , 0       , 0       , dt**2/2 ],
                           [0, 0, 0, 1  , 0  , 0  , dt      , 0       , 0       ],
                           [0, 0, 0, 0  , 1  , 0  , 0       , dt      , 0       ],
                           [0, 0, 0, 0  , 0  , 1  , 0       , 0       , dt      ],
                           [0, 0, 0, 0  , 0  , 0  , 1       , 0       , 0       ],
                           [0, 0, 0, 0  , 0  , 0  , 0       , 1       , 0       ],
                           [0, 0, 0, 0  , 0  , 0  , 0       , 0       , 1       ]])
        
        #Process Noise Matrix for continuous velocity model - Noise in x and y should be independent
        sigA = 0.1
        self.Q = np.zeros((9, 9))
        self.Q[6, 6] = 1
        self.Q[7, 7] = 1
        self.Q[8, 8] = 1
        self.Q = np.dot(self.A, np.dot(self.Q, self.A.T)) * sigA**2

        #Measurement matrix, x,y,z are measured
        self.H = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 1, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 1, 0, 0, 0, 0, 0, 0],])

        
        #Measurement Covariance Matrix
        r = 2.5**2 #0.8 meters seems reasonable
        self.R = np.eye(3, 3) * r
        
    #Prediction step
    def kalman_predict(self):
        #Update State Matrix
        self.X = self.A @ self.X
        #Update Process covariance matrix
        self.P = self.A @ self.P @ self.A.T + self.Q

        return self.X

    #Correction step
    def kalman_correction(self):
        #Measurement input
        pt = self.latest_pose.center_3d
        Y = np.array([pt[0], pt[1], pt[2]]) #Y:(3x1)

        #Kalman Gain
        S = (self.H @ (self.P @ self.H.T)) + self.R
        self.KG = (self.P @ self.H.T) @ np.linalg.inv(S) #KG:(9x3)

        #Update prediction with measurement and Kalman gain
        self.X = self.X + self.KG @ (Y - self.H @ self.X) #X:(9,1), H:(3,9), (Y-H@X): (3,1)

        #Update Process Covariance Matrix
        I = np.eye(9)
        #self.P = (I - self.KG @ self.H) @ self.P #Short but numerically unstable
        self.P = (I - self.KG @ self.H) @ self.P @ (I - self.KG @ self.H).T + self.KG @ self.R @ self.KG.T

        return self.X
    
    def print_kalman(self):
        if not self.kalman:
            return
        print("\n\n\nCorrection Step:\n", not self.predict)
        print("State Vector:\n", self.X)
        print("State Covariance:\n", self.P)
        print("Kalman Gain:\n", self.KG)
        if (self.KG > 1).any():
            print("KG larger than 1???????")
            print("R:\n", self.R)
            # print("Zähler:\n", (self.P @ self.H.T))
            # print("Nenner:\n", np.linalg.pinv(self.H @ self.P @ self.H.T + self.R))
            # print("ZählerDOT:\n", np.dot(self.P, self.H.T))
            # print("NennerDOT:\n", np.linalg.pinv(np.dot(self.H, np.dot(self.P, self.H.T)) + self.R))
            # tS = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R
            # tKG = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(tS))
            # print("TEST KG:\n", tKG)

    def print_kalman_static(self):
        if not self.kalman:
            return
        print("R:\n", self.R)
        print("Q:\n", self.Q)