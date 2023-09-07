import numpy as np
import os
import cv2
import open3d as o3d

class Pose:
    def __init__(self, bb, ego_pos, center_3d=np.nan, center_2d=np.nan, 
                 cluster_center=np.nan, state_vector=np.nan, 
                 P=np.nan, KG=np.nan, ann_center=np.nan):
        self.bb = bb# [x_min, y_min, x_max, y_max]
        self.center_3d = center_3d
        self.center_2d = center_2d
        self.ego_pos = ego_pos #[x, y, z] in meters, from lidar sample_data, z = 0
        self.cluster_center = cluster_center
        self.ann_center = ann_center
        self.state_vector = state_vector
        self.P = P
        self.KG = KG
    
    def reset(self):
        self.center_2d = np.nan
        self.center_3d = np.nan
        self.ego_pos = np.nan
        self.cluster_center = np.nan
        self.ann_center = np.nan
        self.state_vector = np.nan
        self.P = np.nan
        self.KG = np.nan
        #self.bb = np.nan


    