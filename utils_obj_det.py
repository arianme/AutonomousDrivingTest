import numpy as np
import os
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from PIL import Image
import matplotlib.pyplot as plt

from nuscenes.nuscenes import NuScenes
import nuscenes.lidarseg
from nuscenes.utils.data_classes import LidarPointCloud, RadarPointCloud, Box
#import utils
#from utils import view_scene_map, custom_draw_geometry, front_cam_vid, top_down, interactive_vis, visualize_lidar_in_img
from utils_data_viewer import radar_from_file, fuse_radars_in_ego, visualize_radar_in_img
from utils_data_viewer import project_bb_in_img, project_custom_pcd_in_img, fuse_radars_in_ego_sweeps

import time
from scipy.spatial.transform import Rotation
from pyquaternion import Quaternion

import cv2
import open3d as o3d

nusc = ""
COCO_INSTANCE_CATEGORY_NAMES = ['__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']


def vis_obj_det_speed(scene, cam_channel='CAM_FRONT', show_speed=True, show_points=False, record=False):
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
                    cv2.circle(cam_img, (int(np.rint(points[0, i])), int(np.rint(points[1, i]))), 5, [0, 0, 255], -1) 

    


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

#Helper function that returns projected radar points for velocity measurment - For sweep data
def get_radar_in_img_sweeps(cam, radar_set):
    #Load camera image
    cam_file = os.path.join(nusc.dataroot, cam['filename'])
    cam_img = cv2.imread(cam_file)

    #Load Radar scan
    scan, fields = fuse_radars_in_ego_sweeps(radar_set) #18xN
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


#######################################
####################################### PCD Tracking
#Decimate, remove outliers, remove ground plane, and find clusters in pcd
#Returns new pcd with cluster colors and a list of cluster labels
def process_and_cluster(pcd):
    #Use 10cm voxel grid downsampling
    voxel_down_pcd = pcd.voxel_down_sample(voxel_size=0.1)

    #Remove statistical outliers
    noise_removed_pcd, ind = voxel_down_pcd.remove_statistical_outlier(nb_neighbors=20,
                                                    std_ratio=1.5)

    #Segment the ground plane
    _, inliers = noise_removed_pcd.segment_plane(distance_threshold=0.25,
                                        ransac_n=3,
                                        num_iterations=1000)

    #All points except those in the groundplane
    plane_outlier_pcd = noise_removed_pcd.select_by_index(inliers, invert=True)

    #Clustering - Get Labels
    #Distance to neighbors in a cluster
    epsilon = 1#0.8
    #Minimum points of a cluster
    min_pts = 3#3
    labels = np.array(plane_outlier_pcd.cluster_dbscan(eps=epsilon, min_points=min_pts))

    #Labels are numbered
    max_label = labels.max()

    #Assign color to clusters based on colormap
    colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
    colors[labels < 0] = [1, 1, 1, 0]
    plane_outlier_pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
    return plane_outlier_pcd, labels

#BEV of clustered pcd with bounding boxes
def top_down_bb(scene, record=False):
    #Visualize a video sequence of LIDAR data
    first_sample = nusc.get('sample', scene['first_sample_token'])
    last_sample = nusc.get('sample', scene['last_sample_token'])
    sample = first_sample

    #This is rendering loop is used for non blocking visualization
    pcd = o3d.geometry.PointCloud()
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    #Set background to black and reduce surfel size
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0, 0, 0])
    opt.point_size = 2.5 #Original 5

    if record:
        name = scene['name'] + "_top_down_lidar.avi"
        out = None
        

    while sample['token'] != last_sample['token']:
        #Load next scan
        lidar = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        lidar_file = nusc.get_sample_data_path(lidar['token'])
        scan = np.fromfile(lidar_file, dtype=np.float32)
        points = scan.reshape(-1, 5) #x, y, z, intensity, Ring Index

        #Update Geometry
        pcd.points = o3d.utility.Vector3dVector(points[:, :3]) #Only need XYZ

        pcd, labels = process_and_cluster(pcd)

        #Go through all labels (0 - labels.max()+1) and get the number of occurences of that label
        #(labels == i) returns a bool map that is true at the locations where the label is i
        #Use the bool map to index a np array of the pcd points
        clusters = [np.array(pcd.points)[labels == i] for i in range(labels.max()+1)]
        #create list of pointclouds from points in cluster
        pcd_clusters = [o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts)) for pts in clusters]
        #Create list of bounding boxes from points in clusters - Need at least 4 points
        bb_clusters = [o3d.geometry.AxisAlignedBoundingBox.create_from_points(o3d.utility.Vector3dVector(pts)) for pts in clusters if len(pts) > 3]

        for bb in bb_clusters:
            bb.color = [1, 1, 1]

        #Add pcd to list
        bb_clusters.insert(0, pcd)
        ##########################################
        ##########################################
        vis.clear_geometries()
        for geo in bb_clusters:
            vis.add_geometry(geo)

        #Update Visualization
        vis.update_geometry(pcd)
        vis.poll_events()
        vis.update_renderer()

        if record:
            img = (np.asarray(vis.capture_screen_float_buffer(True))*255).astype(np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            if out is None:
                out = cv2.VideoWriter(name, cv2.VideoWriter_fourcc(*'DIVX'), 2, (img.shape[1], img.shape[0]))
            out.write(img)

        #Next Sample
        sample = nusc.get('sample', sample['next'])
        
        #Slow down animation
        time.sleep(0.5)

    vis.destroy_window()
    if record:
        out.release()

#BEV of clustered pcd with bounding boxes using all sensor data
def top_down_bb_sweep(scene, record=False, distance_thresh=100):
    #Visualize a video sequence of LIDAR data
    first_sample = nusc.get('sample', scene['first_sample_token'])
   
    #This is rendering loop is used for non blocking visualization
    pcd = o3d.geometry.PointCloud()
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    #Set background to black and reduce surfel size
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0, 0, 0])
    opt.point_size = 2.5 #Original 5

    if record:
        name = scene['name'] + "_top_down_cluster.avi"
        out = None
        
    lidar = nusc.get('sample_data', first_sample['data']['LIDAR_TOP'])
    while True:
        #Load next scan
        #lidar = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        lidar_file = nusc.get_sample_data_path(lidar['token'])
        scan = np.fromfile(lidar_file, dtype=np.float32)
        points = scan.reshape(-1, 5) #x, y, z, intensity, Ring Index

        #Create pcd from xyz points that are closer than distance_thresh
        pcd.points = o3d.utility.Vector3dVector(points[points[:, 1] <  distance_thresh][:, :3]) 

        pcd, labels = process_and_cluster(pcd)

        #Go through all labels (0 - labels.max()+1) and get the number of occurences of that label
        #(labels == i) returns a bool map that is true at the locations where the label is i
        #Use the bool map to index a np array of the pcd points
        clusters = [np.array(pcd.points)[labels == i] for i in range(labels.max()+1)]
        #create list of pointclouds from points in cluster
        pcd_clusters = [o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts)) for pts in clusters]
        #Create list of bounding boxes from points in clusters - Need at least 4 points
        bb_clusters = [o3d.geometry.AxisAlignedBoundingBox.create_from_points(o3d.utility.Vector3dVector(pts)) for pts in clusters if len(pts) > 3]

        for bb in bb_clusters:
            bb.color = [1, 1, 1]

        #Add pcd to list
        bb_clusters.insert(0, pcd)
        ##########################################
        ##########################################
        vis.clear_geometries()
        for geo in bb_clusters:
            vis.add_geometry(geo)

        #Update Visualization
        vis.update_geometry(pcd)
        vis.poll_events()
        vis.update_renderer()

        if record:
            img = (np.asarray(vis.capture_screen_float_buffer(True))*255).astype(np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            if out is None:
                out = cv2.VideoWriter(name, cv2.VideoWriter_fourcc(*'DIVX'), 8, (img.shape[1], img.shape[0]))
            out.write(img)

        if lidar['next'] == '':
            #lidar = nusc.get('sample_data', first_sample['data']['LIDAR_TOP'])
            break
        else:
            lidar = nusc.get('sample_data', lidar['next'])
        
        #Slow down animation
        #time.sleep(0.5)

    vis.destroy_window()
    if record:
        out.release()

#Visualize clustered points with a bounding box for a single sample
def cluster_sample(sample, record=False):
    pcd = o3d.geometry.PointCloud()

    #Load next scan
    lidar = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
    lidar_file = nusc.get_sample_data_path(lidar['token'])
    scan = np.fromfile(lidar_file, dtype=np.float32)
    points = scan.reshape(-1, 5) #x, y, z, intensity, Ring Index

    pcd.points = o3d.utility.Vector3dVector(points[:, :3])

    pcd, labels = process_and_cluster(pcd)

    #Go through all labels (0 - labels.max()+1) and get the number of occurences of that label
    #(labels == i) returns a bool map that is true at the locations where the label is i
    #Use the bool map to index a np array of the pcd points
    clusters = [np.array(pcd.points)[labels == i] for i in range(labels.max()+1)]
    #create list of pointclouds from points in cluster
    pcd_clusters = [o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts)) for pts in clusters]
    #Create list of bounding boxes from points in clusters - Need at least 4 points
    bb_clusters = [o3d.geometry.AxisAlignedBoundingBox.create_from_points(o3d.utility.Vector3dVector(pts)) for pts in clusters if len(pts) > 3]

    for bb in bb_clusters:
        bb.color = [0, 0, 0]

    #Add pcd to list
    bb_clusters.insert(0, pcd)
    o3d.visualization.draw_geometries(bb_clusters)
    o3d.visualization.draw_geometries([pcd])

#Input is a list of clusters, the image to which they should be projected, the cam sample_data,
#and lidar sample_data
#Returns the image with the bb drawn into it and
#a np array that contains the img points of the bb corners
def draw_bb_points_in_image(clusters, cam_img, cam, lidar):
    #Create list of bounding boxes from points in clusters - Need at least 4 points
    bb_clusters = [o3d.geometry.AxisAlignedBoundingBox.create_from_points(o3d.utility.Vector3dVector(pts)) for pts in clusters if len(pts) > 3]
    #Get corner points of bbs and project them into image
    corner_pts = np.asarray([bb.get_box_points() for bb in bb_clusters]) #(N,8,3)
    corner_pts = corner_pts.reshape(corner_pts.shape[0]*8, 3)
    corner_pts_pcd = o3d.geometry.PointCloud()
    corner_pts_pcd.points = o3d.utility.Vector3dVector(corner_pts)
    corner_pts_img, _ = project_bb_in_img(cam, lidar, corner_pts_pcd) #Result: (3, N)
    corner_pts_img = np.asarray(np.rint(corner_pts_img), dtype=np.int)
    #Draw bb 
    color = (255, 36, 36)
    thickness = 3
    for i in range(0, corner_pts_img.shape[1]-8, 8):
        pt1 = (int(corner_pts_img[0][i]), int(corner_pts_img[1][i]))
        pt2 = (corner_pts_img[0][i+1], corner_pts_img[1][i+1])
        pt3 = (corner_pts_img[0][i+2], corner_pts_img[1][i+2])
        pt4 = (corner_pts_img[0][i+3], corner_pts_img[1][i+3])
        pt5 = (corner_pts_img[0][i+4], corner_pts_img[1][i+4])
        pt6 = (corner_pts_img[0][i+5], corner_pts_img[1][i+5])
        pt7 = (corner_pts_img[0][i+6], corner_pts_img[1][i+6])
        pt8 = (corner_pts_img[0][i+7], corner_pts_img[1][i+7])
        cv2.line(cam_img, pt1, pt2, color, thickness)
        cv2.line(cam_img, pt1, pt3, color, thickness)
        cv2.line(cam_img, pt1, pt4, color, thickness)
        cv2.line(cam_img, pt2, pt7, color, thickness)
        cv2.line(cam_img, pt2, pt8, color, thickness)
        cv2.line(cam_img, pt3, pt6, color, thickness)
        cv2.line(cam_img, pt3, pt8, color, thickness)
        cv2.line(cam_img, pt4, pt6, color, thickness)
        cv2.line(cam_img, pt4, pt7, color, thickness)
        cv2.line(cam_img, pt5, pt6, color, thickness)
        cv2.line(cam_img, pt5, pt7, color, thickness)
        cv2.line(cam_img, pt5, pt8, color, thickness)

    return cam_img, corner_pts_img


def cluster_in_img_sweep(scene, cam_channel='CAM_FRONT', show_bounding_box=True, show_cluster_points=False, show_cluster_center=False, record=False):
    first_sample = nusc.get('sample', scene['first_sample_token'])
    last_sample = nusc.get('sample', scene['last_sample_token'])

    winname = "Lidar - " + cam_channel
    window_width = nusc.get('sample_data', first_sample['data'][cam_channel])['width']
    window_height = nusc.get('sample_data', first_sample['data'][cam_channel])['height']
    cv2.namedWindow(winname, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(winname, window_width, window_height)

    if record:
        name = scene['name'] + "_cluster_img_" + cam_channel + ".avi"
        out = out = cv2.VideoWriter(name, cv2.VideoWriter_fourcc(*'XVID'), 8, (1600, 900)) #cv2.VideoWriter_fourcc(*'MP4V')

    cam = nusc.get('sample_data', first_sample['data'][cam_channel])
    lidar = nusc.get('sample_data', first_sample['data']['LIDAR_TOP'])

    #Get List of all Lidars
    lidars = []
    while lidar['next'] != '':
        lidars.append(lidar)
        lidar = nusc.get('sample_data', lidar['next'])
    #np array of timestamps
    lidar_time = np.asarray([lidar['timestamp'] for lidar in lidars])
    lidar = nusc.get('sample_data', first_sample['data']['LIDAR_TOP'])

    while cam['next'] != '':
        cam_file = os.path.join(nusc.dataroot, cam['filename'])
        cam_img = cv2.imread(cam_file)

        ###################################
        ################################### Visualize Cluster
        #Load lidar scan
        lidar_file = nusc.get_sample_data_path(lidar['token'])
        scan = np.fromfile(lidar_file, dtype=np.float32).reshape(-1, 5) #x, y, z, intensity, Ring Index
        points = scan[:, :3] #Only x, y, z
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        #Create point cloud with cluster colors
        pcd, labels = process_and_cluster(pcd)
        #Project the pcd with its cluster colors into the image
        points, color_img = project_custom_pcd_in_img(cam, lidar, pcd)

        #list of cluster points
        clusters = [np.array(pcd.points)[labels == i] for i in range(labels.max()+1)]
        #Draw bounding box lines into image
        if show_bounding_box:
            cam_img, _ = draw_bb_points_in_image(clusters, cam_img, cam, lidar)
        
        #create list of pointclouds from points in cluster
        pcd_clusters = [o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts)) for pts in clusters]
        #Project center points of bbs into image
        cluster_center = np.array([cl.get_center() for cl in pcd_clusters]) #(N,3)
        cl_center_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(cluster_center))
        cluster_center_img, _ = project_custom_pcd_in_img(cam, lidar, cl_center_pcd)
        cluster_center_img = np.asarray(np.rint(cluster_center_img), dtype=np.int)
        #Draw cluster center dots into image
        if show_cluster_center:
            for i in range(cluster_center_img.shape[1]):
                cv2.circle(cam_img, (cluster_center_img[0, i], cluster_center_img[1, i]), 5, (36, 36, 255), -1) 



        #Draw LIDAR dots into image
        #tolist() brings the values to a python standard type
        if show_cluster_points:
            for i in range(points.shape[1]):
                cv2.circle(cam_img, (int(np.rint(points[0, i])), int(np.rint(points[1, i]))), 2, color_img[i].tolist(), -1) 
        ###################################
        ###################################
        if record:
            out.write(cam_img)

        cv2.imshow(winname, cam_img)
        key = cv2.waitKey(1)
        if key == 32:
            key = 0
            while (key != 32 and key != ord('q') and key != 27): #Space bar
                key = cv2.waitKey(0)

        if key == ord('q') or key == 27:
            break

        #Next Cam Frame
        if cam['next'] == '':
            cam = nusc.get('sample_data', first_sample['data'][cam_channel])
        else:
            cam = nusc.get('sample_data', cam['next'])
        #Find Lidar scan with lowest time diff
        diffs = lidar_time - cam['timestamp']
        diffs = np.where(diffs > 0, diffs, np.inf)
        lidar = lidars[diffs.argmin()]
    cv2.destroyAllWindows()
    if record:
        out.release()


#IoU helper function
#input has shape (N,4)
# X-Axis points right, Y-Axis points down. x_min[0] and y_min[1] are the top left point of bb
def get_IoU(set1, set2):
    #Calculate intersection
    lower_bound = np.maximum(np.expand_dims(set1[:, :2], 1), np.expand_dims(set2[:, :2], 0)) #(N1,N2,2)
    upper_bound = np.minimum(np.expand_dims(set1[:, 2:], 1), np.expand_dims(set2[:, 2:], 0)) #(N1,N2,2)
    #Get the size of the intersecting are, if upper_bound - lower_bound is negative it means that the two boxes do not overlap at all
    #therefore negative intersection sizes get clipped to 0. Otherwise this is the width and height of the intersection
    intersection_size = np.clip(upper_bound - lower_bound, a_min=0, a_max=None) #(N1,N2,2)
    #Calculate intersection area
    intersections = intersection_size[:, :, 0] * intersection_size[:, :, 1] #(N1,N2)

    #Get bb areas for each set
    set1_areas = (set1[:, 2] - set1[:, 0]) * (set1[:, 3] - set1[:, 1]) #(N1)
    set2_areas = (set2[:, 2] - set2[:, 0]) * (set2[:, 3] - set2[:, 1]) #(N2)

    #Calculate union of sets
    unions = np.expand_dims(set1_areas, 1) + np.expand_dims(set2_areas, 0) - intersections #(N1,N2)

    return intersections / unions #(N1,N2)


#Small helper function to determin min max of xyz
def axis_bounds(x_min, x_max, y_min, y_max, z_min, z_max, x, y, z):
    x_min = np.min(x) if np.min(x) < x_min else x_min
    x_max = np.max(x) if np.max(x) > x_max else x_max
    y_min = np.min(y) if np.min(y) < y_min else y_min
    y_max = np.max(y) if np.max(y) > y_max else y_max
    z_min = np.min(z) if np.min(z) < z_min else z_min
    z_max = np.max(z) if np.max(z) > z_max else z_max

    return x_min, x_max, y_min, y_max, z_min, z_max

#Plot pos data in 3D
def plot_3d(pos_p=np.nan, pos_m=np.nan, pos_a=np.nan, scatter=True):
    x = np.empty((0))
    y = np.empty((0))
    z = np.empty((0))

    mea_color = [0.91, 0.59, 0]
    ann_color = [0.6, 0.05, 0]#[0.72, 0.08, 0.5]
    kf_color  = [0, 0.34, 0.57]
    sz=30

    if not np.isnan(pos_p).any():
        x_p = pos_p[:, 0]
        y_p = pos_p[:, 1]
        z_p = pos_p[:, 2]
        
        x = np.concatenate((x, x_p))
        y = np.concatenate((y, y_p))
        z = np.concatenate((z, z_p))
    if not np.isnan(pos_m).any():
        x_m = pos_m[:, 0]
        y_m = pos_m[:, 1]
        z_m = pos_m[:, 2]
        
        x = np.concatenate((x, x_m))
        y = np.concatenate((y, y_m))
        z = np.concatenate((z, z_m))

    if not np.isnan(pos_a).any():
        x_a = pos_a[:, 0]
        y_a = pos_a[:, 1]
        z_a = pos_a[:, 2]
        
        x = np.concatenate((x, x_a))
        y = np.concatenate((y, y_a))
        z = np.concatenate((z, z_a))

    max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() /1.8

    # max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() /2
    mean_x = x.mean()
    mean_y = y.mean()
    mean_z = z.mean()
    x_min = mean_x - max_range
    x_max = mean_x + max_range
    y_min = mean_y - max_range
    y_max = mean_y + max_range
    z_min = mean_z - max_range
    z_max = mean_z + max_range
    
    fig = plt.figure(figsize=(16,9))
    ax = fig.add_subplot(111, projection='3d')
    
    if scatter:
        if not np.isnan(pos_p).any():
            ax.scatter(x_p,y_p,z_p, label='Kalman Filter Estimate', alpha=0.7, color=kf_color, s=sz)
        if not np.isnan(pos_m).any():
            ax.scatter(x_m,y_m,z_m, label='Measurement', alpha=0.7, color=mea_color, s=sz)
        if not np.isnan(pos_a).any():
            ax.scatter(x_a,y_a,z_a, label='Ground Truth', alpha=0.7, color=ann_color, s=sz)
    else:
        if not np.isnan(pos_p).any():
            ax.plot(x_p,y_p,z_p, label='Kalman Filter Estimate')
        if not np.isnan(pos_m).any():
            ax.plot(x_m,y_m,z_m, label='Measurement')
        if not np.isnan(pos_a).any():
            ax.plot(x_a,y_a,z_a, label='Ground Truth')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.title('Car Trajectory estimated with Kalman Filter')

    # Axis equal
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)
    fig.show()

#Plot trajectory in 2D
def plot_2d(pos_p=np.nan, pos_m=np.nan, pos_a=np.nan, scatter=True):
    fig = plt.figure(figsize=(16,9))

    mea_color = [0.91, 0.59, 0]
    ann_color = [0.6, 0.05, 0]#[0.72, 0.08, 0.5]
    kf_color  = [0, 0.34, 0.57]
    sz=30
    
    p_max = 0
    m_max = 0 
    a_max = 0
    if not np.isnan(pos_p).any():
        x_p = pos_p[:, 0]
        y_p = pos_p[:, 1]
        p_max = np.array([x_p.max()-x_p.min(), y_p.max()-y_p.min()]).max()
    if not np.isnan(pos_m).any():
        x_m = pos_m[:, 0]
        y_m = pos_m[:, 1]
        m_max = np.array([x_m.max()-x_m.min(), y_m.max()-y_m.min()]).max() /2
    if not np.isnan(pos_a).any():
        x_a = pos_a[:, 0]
        y_a = pos_a[:, 1]
        a_max = np.array([x_a.max()-x_a.min(), y_a.max()-y_a.min()]).max() /2

    #Axis equal
    if (p_max > m_max) and (p_max > a_max):
        x = x_p
        y = y_p
    elif (m_max > p_max) and (m_max > a_max):
        x = x_m
        y = y_m
    else:
        x = x_a
        y = y_a

    max_range = np.array([x.max()-x.min(), y.max()-y.min()]).max() /2
    mean_x = x.mean()
    mean_y = y.mean()
    x_min = mean_x - max_range
    x_max = mean_x + max_range
    y_min = mean_y - max_range
    y_max = mean_y + max_range
    #plt.plot(x_p, y_p, label='Kalman Filter Estimate')
    #plt.plot(Xr, Zr, label='Real')

    if not np.isnan(pos_p).any():
        plt.scatter(x_p,y_p, label='Kalman Filter Estimate', alpha=0.6, color=kf_color, s=sz)
    if not np.isnan(pos_m).any():
        plt.scatter(x_m,y_m, label='Measurement', alpha=0.6, color=mea_color, s=sz)
    if not np.isnan(pos_a).any():
        plt.scatter(x_a,y_a, label='Ground Truth', alpha=0.6, color=ann_color, s=sz)

    plt.title('Estimate of Car Trajectory')
    plt.legend(loc='best',prop={'size':22})
    plt.axhline(0, color='k')
    plt.axis('equal')
    plt.xlabel('X ($m$)')
    plt.ylabel('Y ($m$)')

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    plt.show()

#Return a np array of all car positions found
def all_car_pos(sample, last_sample):
    #Get All Cars
    all_cars = []
    while sample['token'] != last_sample['token']:
        centers = [nusc.get('sample_annotation', ann_token)['translation'] for ann_token in sample['anns'] if nusc.get('sample_annotation', ann_token)['category_name'] == 'vehicle.car']
        all_cars.extend(centers)
        sample = nusc.get('sample', sample['next'])
    return np.asarray(all_cars)

#Plot 3D and 2D data next to each other
def plot(pos_p=np.nan, pos_m=np.nan, pos_a=np.nan):
    mea_color = [0.91, 0.59, 0]
    ann_color = [0.6, 0.05, 0]#[0.72, 0.08, 0.5]
    kf_color  = [0, 0.34, 0.57]
    sz=30

    x = np.empty((0))
    y = np.empty((0))
    z = np.empty((0))
    if not np.isnan(pos_p).any():
        x_p = pos_p[:, 0]
        y_p = pos_p[:, 1]
        z_p = pos_p[:, 2]
        
        x = np.concatenate((x, x_p))
        y = np.concatenate((y, y_p))
        z = np.concatenate((z, z_p))
    if not np.isnan(pos_m).any():
        x_m = pos_m[:, 0]
        y_m = pos_m[:, 1]
        z_m = pos_m[:, 2]
        
        x = np.concatenate((x, x_m))
        y = np.concatenate((y, y_m))
        z = np.concatenate((z, z_m))

    if not np.isnan(pos_a).any():
        x_a = pos_a[:, 0]
        y_a = pos_a[:, 1]
        z_a = pos_a[:, 2]
        
        x = np.concatenate((x, x_a))
        y = np.concatenate((y, y_a))
        z = np.concatenate((z, z_a))

    max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() /1.8

    # max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() /2
    mean_x = x.mean()
    mean_y = y.mean()
    mean_z = z.mean()
    x_min = mean_x - max_range
    x_max = mean_x + max_range
    y_min = mean_y - max_range
    y_max = mean_y + max_range
    z_min = mean_z - max_range
    z_max = mean_z + max_range

    fig = plt.figure(figsize=(32, 9))
    fig.suptitle('Estimated position')

    #3D plot
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    #ax.plot(x_p,y_p,z_p, label='Kalman Filter Estimate')
    #ax.plot(Xr, Yr, Zr, label='Real')

    if not np.isnan(pos_p).any():
        ax.scatter(x_p,y_p,z_p, label='Kalman Filter Estimate', alpha=0.7, color=kf_color, s=sz)
    if not np.isnan(pos_m).any():
        ax.scatter(x_m,y_m,z_m, label='Measurement', alpha=0.7, color=mea_color, s=sz)
    if not np.isnan(pos_a).any():
        ax.scatter(x_a,y_a,z_a, label='Ground Truth', alpha=0.7, color=ann_color, s=sz)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)
    #plt.title('Car Trajectory estimated with Kalman Filter')

    #2D plot
    ax2 = fig.add_subplot(1, 2, 2)
    #plt.plot(x_p, y_p, label='Kalman Filter Estimate')
    #ax2.scatter(x_p, y_p, label='Kalman Filter Estimate', s=30)
    if not np.isnan(pos_p).any():
        ax2.scatter(x_p,y_p, label='Kalman Filter Estimate', alpha=0.7, color=kf_color, s=sz)
    if not np.isnan(pos_m).any():
        ax2.scatter(x_m,y_m, label='Measurement', alpha=0.7, color=mea_color, s=sz)
    if not np.isnan(pos_a).any():
        ax2.scatter(x_a,y_a, label='Ground Truth', alpha=0.7, color=ann_color, s=sz)

    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.legend()

    ax2.set_xlim(x_min, x_max)
    ax2.set_ylim(y_min, y_max)

    plt.show()







































# #Plot pos data in 3D
# def plot_3d(pos_p=np.nan, pos_m=np.nan, pos_a=np.nan, scatter=True):
#     x_min = np.inf
#     x_max = 0
#     y_min = np.inf
#     y_max = 0
#     z_min = np.inf
#     z_max = 0
    
#     p_max = 0
#     m_max = 0 
#     a_max = 0

#     x = np.empty((0))
#     y = np.empty((0))
#     z = np.empty((0))
#     if not np.isnan(pos_p).any():
#         x_p = pos_p[:, 0]
#         y_p = pos_p[:, 1]
#         z_p = pos_p[:, 2]
#         x_min, x_max, y_min, y_max, z_min, z_max = axis_bounds(x_min, x_max, y_min, y_max, z_min, z_max, x_p, y_p, z_p)
#         x = np.concatenate((x, x_p))
#         y = np.concatenate((y, y_p))
#         z = np.concatenate((z, z_p))

#         #p_max = np.array([x_p.max()-x_p.min(), y_p.max()-y_p.min(), z_p.max()-z_p.min()]).max()
#     if not np.isnan(pos_m).any():
#         x_m = pos_m[:, 0]
#         y_m = pos_m[:, 1]
#         z_m = pos_m[:, 2]
#         x_min, x_max, y_min, y_max, z_min, z_max = axis_bounds(x_min, x_max, y_min, y_max, z_min, z_max, x_m, y_m, z_m)
#         x = np.concatenate((x, x_m))
#         y = np.concatenate((y, y_m))
#         z = np.concatenate((z, z_m))
#         #m_max = np.array([x_m.max()-x_m.min(), y_m.max()-y_m.min(), z_m.max()-z_m.min()]).max() /2
#     if not np.isnan(pos_a).any():
#         x_a = pos_a[:, 0]
#         y_a = pos_a[:, 1]
#         z_a = pos_a[:, 2]
#         x_min, x_max, y_min, y_max, z_min, z_max = axis_bounds(x_min, x_max, y_min, y_max, z_min, z_max, x_a, y_a, z_a)
#         x = np.concatenate((x, x_m))
#         y = np.concatenate((y, y_m))
#         z = np.concatenate((z, z_m))
#         #a_max = np.array([x_a.max()-x_a.min(), y_a.max()-y_a.min(), z_a.max()-z_a.min()]).max() /2
#     #max_range = np.array([x_max-x_min, y_max-y_min, z_max-z_min]).max() /2
#     max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() /2
#     # #Axis equal
#     # if (p_max > m_max) and (p_max > a_max):
#     #     x = x_p
#     #     y = y_p
#     #     z = z_p
#     # elif (m_max > p_max) and (m_max > a_max):
#     #     x = x_m
#     #     y = y_m
#     #     z = z_m
#     # else:
#     #     x = x_a
#     #     y = y_a
#     #     z = z_a

#     # max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() /2
#     mean_x = x.mean()
#     mean_y = y.mean()
#     mean_z = z.mean()
#     x_min = mean_x - max_range
#     x_max = mean_x + max_range
#     y_min = mean_y - max_range
#     y_max = mean_y + max_range
#     z_min = mean_z - max_range
#     z_max = mean_z + max_range
    
#     fig = plt.figure(figsize=(16,9))
#     ax = fig.add_subplot(111, projection='3d')
    
#     if scatter:
#         if not np.isnan(pos_p).any():
#             ax.scatter(x_p,y_p,z_p, label='Kalman Filter Estimate')
#         if not np.isnan(pos_m).any():
#             ax.scatter(x_m,y_m,z_m, label='Measurement')
#         if not np.isnan(pos_a).any():
#             ax.scatter(x_a,y_a,z_a, label='Ground Truth')
#     else:
#         if not np.isnan(pos_p).any():
#             ax.plot(x_p,y_p,z_p, label='Kalman Filter Estimate')
#         if not np.isnan(pos_m).any():
#             ax.plot(x_m,y_m,z_m, label='Measurement')
#         if not np.isnan(pos_a).any():
#             ax.plot(x_a,y_a,z_a, label='Ground Truth')

#     ax.set_xlabel('X')
#     ax.set_ylabel('Y')
#     ax.set_zlabel('Z')
#     ax.legend()
#     plt.title('Car Trajectory estimated with Kalman Filter')

#     # Axis equal
#     ax.set_xlim(x_min, x_max)
#     ax.set_ylim(y_min, y_max)
#     ax.set_zlim(z_min, z_max)
#     fig.show()

##### Same as above but in notebook
#TEEEEEEEEEEEEEEEST
# x_min = np.inf
# x_max = 0
# y_min = np.inf
# y_max = 0
# z_min = np.inf
# z_max = 0

# p_max = 0
# m_max = 0 
# a_max = 0

# x = np.empty((0))
# y = np.empty((0))
# z = np.empty((0))
# if not np.isnan(pos_p).any():
#     x_p = pos_p[:, 0]
#     y_p = pos_p[:, 1]
#     z_p = pos_p[:, 2]
#     #x_min, x_max, y_min, y_max, z_min, z_max = axis_bounds(x_min, x_max, y_min, y_max, z_min, z_max, x_p, y_p, z_p)
#     x = np.concatenate((x, x_p))
#     y = np.concatenate((y, y_p))
#     z = np.concatenate((z, z_p))
#     print("Pos P: ", pos_p.shape)
#     print(x.shape)
#     #p_max = np.array([x_p.max()-x_p.min(), y_p.max()-y_p.min(), z_p.max()-z_p.min()]).max()
# if not np.isnan(pos_m).any():
#     x_m = pos_m[:, 0]
#     y_m = pos_m[:, 1]
#     z_m = pos_m[:, 2]
#     #x_min, x_max, y_min, y_max, z_min, z_max = axis_bounds(x_min, x_max, y_min, y_max, z_min, z_max, x_m, y_m, z_m)
#     x = np.concatenate((x, x_m))
#     y = np.concatenate((y, y_m))
#     z = np.concatenate((z, z_m))
#     print("Pos M: ", pos_m.shape)
#     print(x.shape)
#     #m_max = np.array([x_m.max()-x_m.min(), y_m.max()-y_m.min(), z_m.max()-z_m.min()]).max() /2
# if not np.isnan(pos_a).any():
#     x_a = pos_a[:, 0]
#     y_a = pos_a[:, 1]
#     z_a = pos_a[:, 2]
#     #x_min, x_max, y_min, y_max, z_min, z_max = axis_bounds(x_min, x_max, y_min, y_max, z_min, z_max, x_a, y_a, z_a)
#     x = np.concatenate((x, x_a))
#     y = np.concatenate((y, y_a))
#     z = np.concatenate((z, z_a))
#     print("Pos A: ", pos_a.shape)
#     print(x.shape)
#     #a_max = np.array([x_a.max()-x_a.min(), y_a.max()-y_a.min(), z_a.max()-z_a.min()]).max() /2
# #max_range = np.array([x_max-x_min, y_max-y_min, z_max-z_min]).max() /2
# max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() /2
# # #Axis equal
# # if (p_max > m_max) and (p_max > a_max):
# #     x = x_p
# #     y = y_p
# #     z = z_p
# # elif (m_max > p_max) and (m_max > a_max):
# #     x = x_m
# #     y = y_m
# #     z = z_m
# # else:
# #     x = x_a
# #     y = y_a
# #     z = z_a

# # max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() /2
# mean_x = x.mean()
# mean_y = y.mean()
# mean_z = z.mean()
# x_min = mean_x - max_range
# x_max = mean_x + max_range
# y_min = mean_y - max_range
# y_max = mean_y + max_range
# z_min = mean_z - max_range
# z_max = mean_z + max_range

# fig = plt.figure(figsize=(16,9))
# ax = fig.add_subplot(111, projection='3d')
# scatter = True
# if scatter:
#     if not np.isnan(pos_p).any():
#         ax.scatter(x_p,y_p,z_p, label='Kalman Filter Estimate', alpha=0.5)
#     if not np.isnan(pos_m).any():
#         ax.scatter(x_m,y_m,z_m, label='Measurement', alpha=0.5)
#     if not np.isnan(pos_a).any():
#         ax.scatter(x_a,y_a,z_a, label='Ground Truth')
# else:
#     if not np.isnan(pos_p).any():
#         ax.plot(x_p,y_p,z_p, label='Kalman Filter Estimate')
#     if not np.isnan(pos_m).any():
#         ax.plot(x_m,y_m,z_m, label='Measurement')
#     if not np.isnan(pos_a).any():
#         ax.plot(x_a,y_a,z_a, label='Ground Truth')

# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# ax.legend()
# plt.title('Car Trajectory estimated with Kalman Filter')

# # Axis equal
# ax.set_xlim(x_min, x_max)
# ax.set_ylim(y_min, y_max)
# ax.set_zlim(z_min, z_max)
# fig.show()