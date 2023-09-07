import numpy as np
import os
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from PIL import Image

from nuscenes.nuscenes import NuScenes
import nuscenes.lidarseg
from nuscenes.utils.data_classes import LidarPointCloud, RadarPointCloud, Box
#import utils
#from utils import view_scene_map, custom_draw_geometry, front_cam_vid, top_down, interactive_vis, visualize_lidar_in_img
from utils_data_viewer import radar_from_file, fuse_radars_in_ego, visualize_radar_in_img

import time
from scipy.spatial.transform import Rotation
from pyquaternion import Quaternion

import cv2
import open3d as o3d

nusc = ""
COCO_INSTANCE_CATEGORY_NAMES = ['__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']


def vis_obj_det_speed(scene, cam_channel, show_speed=True, show_points=False, record=False):
    first_sample = nusc.get('sample', scene['first_sample_token'])
    last_sample = nusc.get('sample', scene['last_sample_token'])

    #Load Faster RCNN model, set it to evaluation mode and send it to GPU
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    model.eval()
    model.to('cuda')

    #Confidence threshold, box and text size/thickness
    threshold = 0.8
    rect_th=1
    text_size=0.9
    text_th=2

    if record:
        name = scene['name'] + "_det_speed_" + cam_channel + ".avi"
        out = cv2.VideoWriter(name, cv2.VideoWriter_fourcc(*'DIVX'), 2, (1600, 900))

    #Render Loop
    sample = first_sample
    while sample['token'] != last_sample['token']:
        if sample['token'] == last_sample['token']:
            sample = first_sample
        cam = nusc.get('sample_data', sample['data'][cam_channel])
        cam_file = os.path.join(nusc.dataroot, cam['filename'])
        cam_img = cv2.imread(cam_file)

        #Change to RGB only for conversion to tensor, add dimension for model prediction
        cam_tensor = F.to_tensor(cv2.cvtColor(cam_img, cv2.COLOR_BGR2RGB)).unsqueeze(0).to('cuda')
        
        with torch.no_grad():
            pred = model(cam_tensor)
        
        #Match extracted labels to pre defined labels list
        pred_class = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(pred[0]['labels'].cpu().numpy())]
        #Create tuples from the bounding box for use with cv2.rectangle
        pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].cpu().detach().numpy().astype(int))]
        #Get Scores
        pred_score = list(pred[0]['scores'].cpu().detach().numpy())
        #Get indices of scores above threshold - All results are sorted with decreasing confidence
        cutoff_index = [pred_score.index(x) for x in pred_score if x>threshold]
        #Check if any detections have confidence above threshold
        if len(cutoff_index) > 0:
            #Set pred_t to last element in list, thus last index which has score above threshold
            cutoff_index = cutoff_index[-1]
            #Slice prediction box and labels to only contain values above threshold
            pred_boxes = pred_boxes[:cutoff_index+1]
            pred_class = pred_class[:cutoff_index+1]

            ##########################################
            ##########################################
            #Get velocities
            points, scan = get_radar_in_img(sample, cam_channel)
            #Combined mask of all bbs for visualization
            comb_mask = np.zeros(points.shape[1])
            for i in range(len(pred_boxes)):
                #Get radar points
                xmin = pred_boxes[i][0][0]
                ymin = pred_boxes[i][0][1]
                xmax = pred_boxes[i][1][0]
                ymax = pred_boxes[i][1][1]
                mask = np.ones(points.shape[1])
                mask = np.logical_and(mask, points[0, :] >= xmin)
                mask = np.logical_and(mask, points[0, :] <= xmax)
                mask = np.logical_and(mask, points[1, :] >= ymin)
                mask = np.logical_and(mask, points[1, :] <= ymax)
                comb_mask = np.logical_or(comb_mask, mask)

                #Check if any radar points are inside bb 
                if np.any(mask) and show_speed:
                    #X and Y velocities (compensated) from points in bb
                    obj_v = scan[8:10, mask] #Compensated V - Raw V would be [6:8]
                    #Speed from velocity vector
                    speed = np.linalg.norm(obj_v, axis=0)
                    v_max_index = np.argmax(speed)
                    #Get vx from original radar data
                    vx = obj_v[0, v_max_index]
                    #Fastest lidar scan point with correct sign
                    v_max = speed[v_max_index]*np.sign(vx)
                    
                    text = pred_class[i] + ",v:" + "{:.2f}".format(v_max)
                else: 
                    text = pred_class[i]

                text_p = (pred_boxes[i][0][0], pred_boxes[i][0][1]-5) #10 px above corner
                cv2.rectangle(cam_img, pred_boxes[i][0], pred_boxes[i][1], color=(36,255,12), thickness=rect_th)
                cv2.putText(cam_img, text, text_p, cv2.FONT_HERSHEY_SIMPLEX, text_size, (36,255,12),thickness=text_th)

            points = points[:, comb_mask]
            #Visualize points inside bbs
            if show_points:
                for i in range(points.shape[1]):
                    cv2.circle(cam_img, (int(np.rint(points[0, i])), int(np.rint(points[1, i]))), 2, [0, 0, 255], -1) 

    


            ##########################################
            ##########################################


            # #Draw bounding boxes
            # for i in range(len(pred_boxes)):
            #     cv2.rectangle(cam_img, pred_boxes[i][0], pred_boxes[i][1], color=(0, 255, 0), thickness=rect_th)
            #     cv2.putText(cam_img, pred_class[i], pred_boxes[i][0], cv2.FONT_HERSHEY_SIMPLEX, text_size, (0,255,0),thickness=text_th)

        if record:
            out.write(cam_img)

        winname = cam['channel']
        window_width, window_height = 1600, 900
        cv2.namedWindow(winname, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(winname, window_width, window_height)

        cv2.imshow(winname, cam_img)
        key = cv2.waitKey(250)
        if key == 32:
            key = 0
            while (key != 32 and key != ord('q') and key != 27): #Space bar
                key = cv2.waitKey(0)

        if key == ord('q') or key == 27:
            break

        sample = nusc.get('sample', sample['next'])
    cv2.destroyAllWindows()

    if record:
        out.release()

#Helper function that returns the projected radar points in the image
def get_radar_in_img(sample, cam_channel='CAM_FRONT'):
    #Load camera image
    cam = nusc.get('sample_data', sample['data'][cam_channel])
    cam_file = os.path.join(nusc.dataroot, cam['filename'])
    cam_img = cv2.imread(cam_file)

    #Load Radar scan
    scan, fields = fuse_radars_in_ego(sample) #scan 18 x N
    points = scan.transpose()[:, :3] #Does not need to be converted to int - Needs to be Nx18 for o3d

    #Get Camera Pose and intrinsic parameters
    cam_sensor = nusc.get('calibrated_sensor', cam['calibrated_sensor_token'])
    cam_t = np.asarray(cam_sensor['translation'])
    cam_r_quat = Quaternion(cam_sensor['rotation'])
    cam_k = np.asarray(cam_sensor['camera_intrinsic'])

    #Transform points to image frame - Using Open3D transformations
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    #Second transformation - From ego to camera frame
    pcd.translate(-cam_t)
    pcd.rotate(cam_r_quat.rotation_matrix.T, np.array([0, 0, 0]))

    #Project points into image plane
    view = np.copy(cam_k)
    points = np.asarray(pcd.points).T[:3, :]

    #Used to filter points
    depth = points[2, :]

    #Prepare intrinsics matrix for homogenous transformation
    viewpad = np.eye(4)
    viewpad[:view.shape[0], :view.shape[1]] = view

    #Prepare points for homogenous transformation
    points = np.concatenate((points, np.ones((1, points.shape[1])))) #4 X N matrix - Homogenous

    #Project Points into image
    points = np.dot(viewpad, points)

    #Remove w
    points = points[:3, :]

    #Normalize along Z axis - Divide by depth
    points = points / points[2, :]

    # Remove points that are either outside or behind the camera. Leave a margin of 1 pixel for aesthetic reasons.
    # Also make sure points are at least 1m in front of the camera to avoid seeing the lidar points on the camera
    # casing for non-keyframes which are slightly out of sync.
    min_dist = 1.0
    mask = np.ones(depth.shape[0], dtype=bool)
    mask = np.logical_and(mask, depth > min_dist)
    mask = np.logical_and(mask, points[0, :] > 1)
    mask = np.logical_and(mask, points[0, :] < cam_img.shape[1] - 1)#cv image has x axis on dimension 1
    mask = np.logical_and(mask, points[1, :] > 1)
    mask = np.logical_and(mask, points[1, :] < cam_img.shape[0] - 1)
    points = points[:2, mask] #Z is 1 anyway


    #Return the projected points, as well as the filtered original radar data
    return points, scan[:, mask]



