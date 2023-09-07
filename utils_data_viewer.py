import cv2
import os
import open3d as o3d
from nuscenes.nuscenes import NuScenes
import numpy as np
import time
from pyquaternion import Quaternion
import struct
import matplotlib.pyplot as plt

nusc = ""

def view_scene_map(scene):
    if nusc == "":
        print("Error! utils.nusc not set!")
        return 
    #Show the map of the scene
    log = nusc.get('log', scene['log_token'])
    map = nusc.get('map', log['map_token'])
    map_file = os.path.join(nusc.dataroot, map['filename'])
    map_img = cv2.imread(map_file)

    winname = 'Map from ' + scene['name']
    window_width, window_height = 800, 600
    cv2.namedWindow(winname, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(winname, window_width, window_height)

    cv2.imshow(winname, map_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def custom_draw_geometry(pcd):
    # The following code achieves the same effect as:
    # o3d.visualization.draw_geometries([pcd])
    vis = o3d.visualization.Visualizer()
    vis.create_window()
        
    #Set background to black
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0, 0, 0])
    opt.point_size = 2.5 #Original 5

    vis.add_geometry(pcd)
    vis.run()
    vis.destroy_window()

#Visualize raw camera feed from a selected scene and channel
def cam_vis(scene, cam_channel, record=False):
    first_sample = nusc.get('sample', scene['first_sample_token'])
    last_sample = nusc.get('sample', scene['last_sample_token'])

    if record:
        name = scene['name'] + "_cam_" + cam_channel + ".avi"
        out = out = cv2.VideoWriter(name, cv2.VideoWriter_fourcc(*'XVID'), 2, (1600, 900)) #cv2.VideoWriter_fourcc(*'MP4V')

    sample = first_sample
    while sample['token'] != last_sample['token']:
        if sample['token'] == last_sample['token']:
            sample = first_sample
        cam = nusc.get('sample_data', sample['data'][cam_channel])
        cam_file = os.path.join(nusc.dataroot, cam['filename'])
        cam_img = cv2.imread(cam_file)

        if record:
            out.write(cam_img)

        winname = cam['channel']
        window_width, window_height = 800, 600
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

def cam_vis_sweep(scene, cam_channel, record=False):
    first_sample = nusc.get('sample', scene['first_sample_token'])
    last_sample = nusc.get('sample', scene['last_sample_token'])

    if record:
        name = scene['name'] + "_cam_" + cam_channel + ".avi"
        out = out = cv2.VideoWriter(name, cv2.VideoWriter_fourcc(*'XVID'), 2, (1600, 900)) #cv2.VideoWriter_fourcc(*'MP4V')

    cam = nusc.get('sample_data', first_sample['data'][cam_channel])
    while cam['next'] != '':
        cam_file = os.path.join(nusc.dataroot, cam['filename'])
        cam_img = cv2.imread(cam_file)

        if record:
            out.write(cam_img)

        winname = cam['channel']
        window_width, window_height = 800, 600
        cv2.namedWindow(winname, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(winname, window_width, window_height)

        cv2.imshow(winname, cam_img)
        key = cv2.waitKey(45)
        if key == 32:
            key = 0
            while (key != 32 and key != ord('q') and key != 27): #Space bar
                key = cv2.waitKey(0)

        if key == ord('q') or key == 27:
            break

        cam = nusc.get('sample_data', cam['next'])
    cv2.destroyAllWindows()
    if record:
        out.release()

#Use Open3D to visualize a fixed top down view on lidar throughout scene
def top_down(scene, record=False):
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
        vis.clear_geometries()
        vis.add_geometry(pcd)

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
        time.sleep(0.25)

    vis.destroy_window()
    if record:
        out.release()

def interactive_vis(scene, record=False):
    first_sample = nusc.get('sample', scene['first_sample_token'])
    last_sample = nusc.get('sample', scene['last_sample_token'])
    sample = first_sample
    pcd = o3d.geometry.PointCloud()

    if record:
        name = scene['name'] + "_interactive_lidar.avi"
        out = None 
 
    #Render callback for non blocking visualization which allows smoother control of view
    #Define the callback here to simplify the transfer of variables samples, and pcd
    def render_callback(vis):
        nonlocal sample, name, out
        if sample['token'] == last_sample['token']:
            return False

        #Load next scan
        lidar = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        lidar_file = nusc.get_sample_data_path(lidar['token'])
        scan = np.fromfile(lidar_file, dtype=np.float32)
        points = scan.reshape(-1, 5) #x, y, z, intensity, Ring Index

        #Update Geometry
        if sample['token'] == first_sample['token']:
            pcd.points = o3d.utility.Vector3dVector(points[:, :3])
            vis.add_geometry(pcd)
        pcd.points = o3d.utility.Vector3dVector(points[:, :3]) #Only need XYZ
        #vis.clear_geometries()
        #vis.add_geometry(pcd)

        # Update the visualization window and process events to handle keyboard inputs.
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
        time.sleep(0.25)

        return True

    
    # Create an Open3D visualization window.
    vis = o3d.visualization.VisualizerWithKeyCallback()

    # Set the custom update function.
    vis.register_animation_callback(render_callback)
    

    # Start animation.
    vis.create_window()

    opt = vis.get_render_option()
    opt.background_color = np.asarray([0, 0, 0])
    opt.point_size = 2.5 #Original 5

    vis.run()
    vis.destroy_window()
    if record:
        out.release()

#Show video of selected camera with lidar rendered into the image - Keyframes only
def visualize_lidar_in_img(scene, cam_channel='CAM_FRONT', record=False):
    first_sample = nusc.get('sample', scene['first_sample_token'])
    last_sample = nusc.get('sample', scene['last_sample_token'])

    winname = "Lidar - " + cam_channel
    window_width = nusc.get('sample_data', first_sample['data'][cam_channel])['width']
    window_height = nusc.get('sample_data', first_sample['data'][cam_channel])['height']
    cv2.namedWindow(winname, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(winname, window_width, window_height)

    if record:
        name = scene['name'] + "_lidar_in_img_" + cam_channel + ".avi"
        out = cv2.VideoWriter(name, cv2.VideoWriter_fourcc(*'DIVX'), 2, (window_width, window_height))

    #Load lidar and img data from scan
    sample = first_sample

    while sample['token'] != last_sample['token']:
        if sample['token'] == last_sample['token']:
            sample = first_sample
        #Load camera image
        cam = nusc.get('sample_data', sample['data'][cam_channel])
        cam_file = os.path.join(nusc.dataroot, cam['filename'])
        cam_img = cv2.imread(cam_file)

        #Load lidar scan
        lidar = nusc.get('sample_data', sample['data']['LIDAR_TOP'])

        _, _, points, color_img = project_points_in_img(cam, lidar)


        #Draw LIDAR dots into image
        #tolist() brings the values to a python standard type
        for i in range(points.shape[1]):
            cv2.circle(cam_img, (int(np.rint(points[0, i])), int(np.rint(points[1, i]))), 2, color_img[i].tolist(), -1) 


        if record:
            out.write(cam_img)

        cv2.imshow(winname, cam_img)
        key = cv2.waitKey(400)
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

#Show video of selected camera with lidar rendered into the image
def visualize_lidar_in_img_sweeps(scene, cam_channel='CAM_FRONT', record=False):
    first_sample = nusc.get('sample', scene['first_sample_token'])
    last_sample = nusc.get('sample', scene['last_sample_token'])

    winname = "Lidar - " + cam_channel
    window_width = nusc.get('sample_data', first_sample['data'][cam_channel])['width']
    window_height = nusc.get('sample_data', first_sample['data'][cam_channel])['height']
    cv2.namedWindow(winname, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(winname, window_width, window_height)

    if record:
        name = scene['name'] + "_lidar_in_img_" + cam_channel + ".avi"
        out = cv2.VideoWriter(name, cv2.VideoWriter_fourcc(*'DIVX'), 2, (window_width, window_height))

    #Load lidar and img data from scan
    sample = first_sample
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

    while True:
        #Load camera image
        cam_file = os.path.join(nusc.dataroot, cam['filename'])
        cam_img = cv2.imread(cam_file)

        _, _, points, color_img = project_points_in_img(cam, lidar)

        #Draw LIDAR dots into image
        #tolist() brings the values to a python standard type
        for i in range(points.shape[1]):
            cv2.circle(cam_img, (int(np.rint(points[0, i])), int(np.rint(points[1, i]))), 2, color_img[i].tolist(), -1) 

        if record:
            out.write(cam_img)

        cv2.imshow(winname, cam_img)
        key = cv2.waitKey(45)
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
        #Replace negative numbers with large positive number
        #LiDAR timestamp is from end of sweep, sweep takes 1/20 second, therefore a lidar timestamp that is older than the cam timestamp
        #almost guarantees that the lidar data is lagging behind. Thus I take the lidar with a timestamp that is later(bigger) 
        #than the cam timestamp --> Difference needs to be positive
        diffs = np.where(diffs > 0, diffs, np.inf)
        lidar = lidars[diffs.argmin()]

    cv2.destroyAllWindows()
    if record:
        out.release()

#Return the projected image coordinates of pcd points
# and color of projected points - Either with original pcd colors or using the JET colormap based on distance
#Arguments are camera sample_data, the pointsensor sample_data, o3d pcd which might contain colors
#Helper function for projecting points from lidar into camera view
def project_custom_pcd_in_img(cam, lidar, pcd):
    #Load Image
    cam_file = os.path.join(nusc.dataroot, cam['filename'])
    cam_img = cv2.imread(cam_file)

    #Load lidar scan
    lidar_file = nusc.get_sample_data_path(lidar['token'])
    #scan = np.fromfile(lidar_file, dtype=np.float32).reshape(-1, 5) #x, y, z, intensity, Ring Index
    #points = scan[:, :3] #Only x, y, z
    points = np.asarray(pcd.points)

    #Get Lidar Pose
    lidar_sensor = nusc.get('calibrated_sensor', lidar['calibrated_sensor_token'])
    lidar_t = np.asarray(lidar_sensor['translation'])
    lidar_r_quat = Quaternion(lidar_sensor['rotation'])
    #Ego pose at lidar timestamp
    lidar_ego = nusc.get('ego_pose', lidar['ego_pose_token'])
    lidar_ego_t = np.asarray(lidar_ego['translation'])
    lidar_ego_r_quat = Quaternion(lidar_ego['rotation'])

    #Get Camera Pose and intrinsic parameters
    cam_sensor = nusc.get('calibrated_sensor', cam['calibrated_sensor_token'])
    cam_t = np.asarray(cam_sensor['translation'])
    cam_r_quat = Quaternion(cam_sensor['rotation'])
    cam_k = np.asarray(cam_sensor['camera_intrinsic'])
    #Ego pose at camera timestamp
    cam_ego = nusc.get('ego_pose', cam['ego_pose_token'])
    cam_ego_t = np.asarray(cam_ego['translation'])
    cam_ego_r_quat = Quaternion(cam_ego['rotation'])

    #Transform points to image frame
    #First transformation - Lidar to Ego
    points = np.dot(lidar_r_quat.rotation_matrix, points.T).T
    points = points + lidar_t
    #Ego to global
    points = np.dot(lidar_ego_r_quat.rotation_matrix, points.T).T
    points = points + lidar_ego_t

    #Global to cam ego
    points = points - cam_ego_t
    points = np.dot(cam_ego_r_quat.rotation_matrix.T, points.T).T
    #Second transformation - Ego to Camera
    points = points - cam_t
    points = np.dot(cam_r_quat.rotation_matrix.T, points.T).T
    points = points.T #Get the points to 3xN

    #Project points into image plane
    view = np.copy(cam_k)

    #Set color to depth value
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
    mask = np.ones(points.shape[1], dtype=bool)
    mask = np.logical_and(mask, depth > min_dist)
    mask = np.logical_and(mask, points[0, :] > 1)
    mask = np.logical_and(mask, points[0, :] < cam_img.shape[1] - 1)#cv image has x axis on dimension 1
    mask = np.logical_and(mask, points[1, :] > 1)
    mask = np.logical_and(mask, points[1, :] < cam_img.shape[0] - 1)
    
    points = points[:, mask]

    #If pcd has colors, use those and filter them by the mask and scale to 255 for visualization - Shape N,3
    if len(pcd.colors) > 0:
        color_img = np.asarray(pcd.colors)[mask, :]*255
    else:
        color_img = np.copy(depth)
        color_img = color_img[mask]
        #Normalize color scalars
        color_img = 1 - (color_img - np.min(color_img)) / np.max(color_img) - np.min(color_img)

        color_map = cv2.COLORMAP_JET
        color_img = cv2.applyColorMap(np.uint8(color_img * 255), color_map).squeeze()

    return points, color_img

#Special helper that considers that a bounding box has eight points that belong together
#If one point does not get projected, none of them will
def project_bb_in_img(cam, lidar, pcd):
    #Load Image
    cam_file = os.path.join(nusc.dataroot, cam['filename'])
    cam_img = cv2.imread(cam_file)

    #Load lidar scan
    lidar_file = nusc.get_sample_data_path(lidar['token'])
    #scan = np.fromfile(lidar_file, dtype=np.float32).reshape(-1, 5) #x, y, z, intensity, Ring Index
    #points = scan[:, :3] #Only x, y, z
    points = np.asarray(pcd.points)

    #Get Lidar Pose
    lidar_sensor = nusc.get('calibrated_sensor', lidar['calibrated_sensor_token'])
    lidar_t = np.asarray(lidar_sensor['translation'])
    lidar_r_quat = Quaternion(lidar_sensor['rotation'])
    #Ego pose at lidar timestamp
    lidar_ego = nusc.get('ego_pose', lidar['ego_pose_token'])
    lidar_ego_t = np.asarray(lidar_ego['translation'])
    lidar_ego_r_quat = Quaternion(lidar_ego['rotation'])

    #Get Camera Pose and intrinsic parameters
    cam_sensor = nusc.get('calibrated_sensor', cam['calibrated_sensor_token'])
    cam_t = np.asarray(cam_sensor['translation'])
    cam_r_quat = Quaternion(cam_sensor['rotation'])
    cam_k = np.asarray(cam_sensor['camera_intrinsic'])
    #Ego pose at camera timestamp
    cam_ego = nusc.get('ego_pose', cam['ego_pose_token'])
    cam_ego_t = np.asarray(cam_ego['translation'])
    cam_ego_r_quat = Quaternion(cam_ego['rotation'])

    #Transform points to image frame
    #First transformation - Lidar to Ego
    points = np.dot(lidar_r_quat.rotation_matrix, points.T).T
    points = points + lidar_t
    #Ego to global
    points = np.dot(lidar_ego_r_quat.rotation_matrix, points.T).T
    points = points + lidar_ego_t

    #Global to cam ego
    points = points - cam_ego_t
    points = np.dot(cam_ego_r_quat.rotation_matrix.T, points.T).T
    #Second transformation - Ego to Camera
    points = points - cam_t
    points = np.dot(cam_r_quat.rotation_matrix.T, points.T).T
    points = points.T #Get the points to 3xN

    #Project points into image plane
    view = np.copy(cam_k)

    #Set color to depth value
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
    mask = np.ones(points.shape[1], dtype=bool)
    mask = np.logical_and(mask, depth > min_dist)
    mask = np.logical_and(mask, points[0, :] > 1)
    mask = np.logical_and(mask, points[0, :] < cam_img.shape[1] - 1)#cv image has x axis on dimension 1
    mask = np.logical_and(mask, points[1, :] > 1)
    mask = np.logical_and(mask, points[1, :] < cam_img.shape[0] - 1)
    
    #If one of the 8 points of a bb is excluded, exclude all eight
    mask = mask.reshape(-1, 8)
    mask_rows = np.any(mask == False, axis=1)
    mask[mask_rows] = False
    mask = mask.flatten()

    points = points[:, mask]

    #If pcd has colors, use those and filter them by the mask and scale to 255 for visualization - Shape N,3
    if len(pcd.colors) > 0:
        color_img = np.asarray(pcd.colors)[mask, :]*255
    else:
        color_img = np.copy(depth)
        color_img = color_img[mask]
        #Normalize color scalars
        color_img = 1 - (color_img - np.min(color_img)) / np.max(color_img) - np.min(color_img)

        color_map = cv2.COLORMAP_JET
        color_img = cv2.applyColorMap(np.uint8(color_img * 255), color_map).squeeze()

    return points, color_img


#Return the original lidar points, color from image for o3d pcd, projected image coordinates of points in a pointcloud
# and color of projected points for opencv using the JET colormap 
#Arguments are 'sample_data' from camera and lidar
#Helper function for projecting points from lidar into camera view
def project_points_in_img(cam, lidar, pcd=None):
    #Load Image
    cam_file = os.path.join(nusc.dataroot, cam['filename'])
    cam_img = cv2.imread(cam_file)

    if pcd == None:
        #Load lidar scan
        lidar_file = nusc.get_sample_data_path(lidar['token'])
        scan = np.fromfile(lidar_file, dtype=np.float32).reshape(-1, 5) #x, y, z, intensity, Ring Index
        points = scan[:, :3] #Only x, y, z
    else:
        points = np.asarray(pcd.points)
        scan = np.copy(points) #Later on I need the unchanged points

    #Get Lidar Pose
    lidar_sensor = nusc.get('calibrated_sensor', lidar['calibrated_sensor_token'])
    lidar_t = np.asarray(lidar_sensor['translation'])
    lidar_r_quat = Quaternion(lidar_sensor['rotation'])
    #Ego pose at lidar timestamp
    lidar_ego = nusc.get('ego_pose', lidar['ego_pose_token'])
    lidar_ego_t = np.asarray(lidar_ego['translation'])
    lidar_ego_r_quat = Quaternion(lidar_ego['rotation'])

    #Get Camera Pose and intrinsic parameters
    cam_sensor = nusc.get('calibrated_sensor', cam['calibrated_sensor_token'])
    cam_t = np.asarray(cam_sensor['translation'])
    cam_r_quat = Quaternion(cam_sensor['rotation'])
    cam_k = np.asarray(cam_sensor['camera_intrinsic'])
    #Ego pose at camera timestamp
    cam_ego = nusc.get('ego_pose', cam['ego_pose_token'])
    cam_ego_t = np.asarray(cam_ego['translation'])
    cam_ego_r_quat = Quaternion(cam_ego['rotation'])

    #Transform points to image frame - Using Open3D transformations
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(points)

    #First transformation
    # pcd.rotate(lidar_r_quat.rotation_matrix, np.array([0, 0, 0]))
    # pcd.translate(lidar_t)
    points = np.dot(lidar_r_quat.rotation_matrix, points.T).T
    points = points + lidar_t
    #Ego to global
    points = np.dot(lidar_ego_r_quat.rotation_matrix, points.T).T
    points = points + lidar_ego_t

    #Global to cam ego
    points = points - cam_ego_t
    points = np.dot(cam_ego_r_quat.rotation_matrix.T, points.T).T

    #Second transformation
    # pcd.translate(-cam_t)
    # pcd.rotate(cam_r_quat.rotation_matrix.T, np.array([0, 0, 0]))
    points = points - cam_t
    points = np.dot(cam_r_quat.rotation_matrix.T, points.T).T
    points = points.T #Get the points to 3xN

    #Project points into image plane
    view = np.copy(cam_k)
    # points = np.asarray(pcd.points).T[:3, :]

    #Set color to depth value
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
    mask = np.ones(points.shape[1], dtype=bool)
    mask = np.logical_and(mask, depth > min_dist)
    mask = np.logical_and(mask, points[0, :] > 1)
    mask = np.logical_and(mask, points[0, :] < cam_img.shape[1] - 1)#cv image has x axis on dimension 1
    mask = np.logical_and(mask, points[1, :] > 1)
    mask = np.logical_and(mask, points[1, :] < cam_img.shape[0] - 1)
    points = points[:, mask]
    pcd_points = scan[mask, :3]

    #Extract color values from image
    pts_color = np.zeros((points.shape[1], 3))
    for i in range(points.shape[1]):
        pts_color[i] = cam_img[int(np.rint(points[1, i])), int(np.rint(points[0, i]))]
    #pts_color = (pts_color - pts_color.min()) / (pts_color.max() - pts_color.min()) #Normalize to range 0 to 1

    #BGR to RGB
    pts_color = pts_color[:, ::-1]
    pcd_color = pts_color.astype(np.float32) / 255.0

    if len(pcd.colors) > 0:
        color_img = np.asarray(pcd.colors)[mask, :]*255
    else:
        color_img = np.copy(depth)
        color_img = color_img[mask]
        if color_img.size > 0:
            #Normalize color scalars
            color_img = 1 - (color_img - np.min(color_img)) / np.max(color_img) - np.min(color_img)

            color_map = cv2.COLORMAP_JET
            color_img = cv2.applyColorMap(np.uint8(color_img * 255), color_map).squeeze()

    return pcd_points, pcd_color, points, color_img
    
def colorise_pcd(scene, record=False):
    first_sample = nusc.get('sample', scene['first_sample_token'])
    last_sample = nusc.get('sample', scene['last_sample_token'])
    sample = first_sample
    channels = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']

    pcd = o3d.geometry.PointCloud()
    
    if record:
        name = scene['name'] + "_colorised_pcd.avi"
        out = None

    #Render callback for non blocking visualization which allows smoother control of view
    #Define the callback here to simplify the transfer of variables samples, and pcd
    def render_callback(vis):
        nonlocal sample, pcd, channels, name, out
        if sample['token'] == last_sample['token']:
            return False

        #Load next scan
        #cam = nusc.get('sample_data', sample['data'][cam_channel])
        #Load lidar scan
        lidar = nusc.get('sample_data', sample['data']['LIDAR_TOP'])

        #Merged points
        pcd_points =np.empty((0, 3))
        pcd_colors =np.empty((0, 3))
        for channel in channels:
            #Load next scan
            cam = nusc.get('sample_data', sample['data'][channel])
            points, color, _, _ = project_points_in_img(cam, lidar)
            pcd_points = np.append(pcd_points, points, axis=0)
            pcd_colors = np.append(pcd_colors, color, axis=0)

        #pcd_points, pcd_color = project_points_in_img(cam, lidar)

        #Update Geometry
        if sample['token'] == first_sample['token']:
            pcd.points = o3d.utility.Vector3dVector(pcd_points)
            pcd.colors = o3d.utility.Vector3dVector(pcd_colors)
            vis.add_geometry(pcd)
        pcd.points = o3d.utility.Vector3dVector(pcd_points) #Only need XYZ
        pcd.colors = o3d.utility.Vector3dVector(pcd_colors)
        #vis.clear_geometries()
        #vis.add_geometry(pcd)

        # Update the visualization window and process events to handle keyboard inputs.
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
        time.sleep(0.25)

        return True


    # Create an Open3D visualization window.
    vis = o3d.visualization.VisualizerWithKeyCallback()

    # Set the custom update function.
    vis.register_animation_callback(render_callback)


    # Start animation.
    vis.create_window()

    opt = vis.get_render_option()
    opt.background_color = np.asarray([0, 0, 0])
    opt.point_size = 2.5 #Original 5

    vis.run()
    vis.destroy_window()
    if record:
        out.release()

##################################################################################
##################################################################################
#RADAR

#This is basically just a copy from the radar pointcloud code from nuscenes devkit
def radar_from_file(radar_file):

    """
    Loads RADAR data from a Point Cloud Data file. See details below.
    :param file_name: The path of the pointcloud file.
    :param invalid_states: Radar states to be kept. See details below.
    :param dynprop_states: Radar states to be kept. Use [0, 2, 6] for moving objects only. See details below.
    :param ambig_states: Radar states to be kept. See details below.
    To keep all radar returns, set each state filter to range(18).
    :return: <np.float: d, n>. Point cloud matrix with d dimensions and n points.

    Example of the header fields:
    # .PCD v0.7 - Point Cloud Data file format
    VERSION 0.7
    FIELDS x y z dyn_prop id rcs vx vy vx_comp vy_comp is_quality_valid ambig_state x_rms y_rms invalid_state pdh0 vx_rms vy_rms
    SIZE 4 4 4 1 2 4 4 4 4 4 1 1 1 1 1 1 1 1
    TYPE F F F I I F F F F F I I I I I I I I
    COUNT 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
    WIDTH 125
    HEIGHT 1
    VIEWPOINT 0 0 0 1 0 0 0
    POINTS 125
    DATA binary

    Below some of the fields are explained in more detail:

    x is front, y is left

    vx, vy are the velocities in m/s.
    vx_comp, vy_comp are the velocities in m/s compensated by the ego motion.
    We recommend using the compensated velocities.

    invalid_state: state of Cluster validity state.
    (Invalid states)
    0x01	invalid due to low RCS
    0x02	invalid due to near-field artefact
    0x03	invalid far range cluster because not confirmed in near range
    0x05	reserved
    0x06	invalid cluster due to high mirror probability
    0x07	Invalid cluster because outside sensor field of view
    0x0d	reserved
    0x0e	invalid cluster because it is a harmonics
    (Valid states)
    0x00	valid
    0x04	valid cluster with low RCS
    0x08	valid cluster with azimuth correction due to elevation
    0x09	valid cluster with high child probability
    0x0a	valid cluster with high probability of being a 50 deg artefact
    0x0b	valid cluster but no local maximum
    0x0c	valid cluster with high artefact probability
    0x0f	valid cluster with above 95m in near range
    0x10	valid cluster with high multi-target probability
    0x11	valid cluster with suspicious angle

    dynProp: Dynamic property of cluster to indicate if is moving or not.
    0: moving
    1: stationary
    2: oncoming
    3: stationary candidate
    4: unknown
    5: crossing stationary
    6: crossing moving
    7: stopped

    ambig_state: State of Doppler (radial velocity) ambiguity solution.
    0: invalid
    1: ambiguous
    2: staggered ramp
    3: unambiguous
    4: stationary candidates

    pdh0: False alarm probability of cluster (i.e. probability of being an artefact caused by multipath or similar).
    0: invalid
    1: <25%
    2: 50%
    3: 75%
    4: 90%
    5: 99%
    6: 99.9%
    7: <=100%
    """
    #RADAR - Read pcd from nuscenes devkit code
    meta = []
    with open(radar_file, 'rb') as f:
        for line in f:
            line = line.strip().decode('utf-8')
            meta.append(line)
            if line.startswith('DATA'):
                break

        data_binary = f.read()

     # Get the header rows and check if they appear as expected.
    assert meta[0].startswith('#'), 'First line must be comment'
    assert meta[1].startswith('VERSION'), 'Second line must be VERSION'
    #Get the fields of the data
    fields = meta[2].split(' ')[1:]
    sizes = meta[3].split(' ')[1:]
    types = meta[4].split(' ')[1:]
    counts = meta[5].split(' ')[1:]
    width = int(meta[6].split(' ')[1])
    height = int(meta[7].split(' ')[1])
    data = meta[10].split(' ')[1]
    feature_count = len(types)
    assert width > 0
    assert len([c for c in counts if c != c]) == 0, 'Error: COUNT not supported!'
    assert height == 1, 'Error: height != 0 not supported!'
    assert data == 'binary'

    types# Lookup table for how to decode the binaries.
    unpacking_lut = {'F': {2: 'e', 4: 'f', 8: 'd'},
                        'I': {1: 'b', 2: 'h', 4: 'i', 8: 'q'},
                        'U': {1: 'B', 2: 'H', 4: 'I', 8: 'Q'}}
    types_str = ''.join([unpacking_lut[t][int(s)] for t, s in zip(types, sizes)])

    # Decode each point.
    offset = 0
    point_count = width
    points = []
    for i in range(point_count):
        point = []
        for p in range(feature_count):
            start_p = offset
            end_p = start_p + int(sizes[p])
            assert end_p < len(data_binary)
            point_p = struct.unpack(types_str[p], data_binary[start_p:end_p])[0]
            point.append(point_p)
            offset = end_p
        points.append(point)

    # A NaN in the first point indicates an empty pointcloud.
    point = np.array(points[0])
    if np.any(np.isnan(point)):
        return np.zeros((feature_count, 0)), ['']

    # Convert to numpy matrix.
    points = np.array(points).transpose()

    # Class-level settings for radar pointclouds, see from_file().
    invalid_states = [0]  # type: List[int]
    dynprop_states = range(7)  # type: List[int] # Use [0, 2, 6] for moving objects only.
    ambig_states = [3]  # type: List[int]

     # If no parameters are provided, use default settings.
    invalid_states = p_invalid_states if invalid_states is None else invalid_states
    dynprop_states = p_dynprop_states if dynprop_states is None else dynprop_states
    ambig_states = p_ambig_states if ambig_states is None else ambig_states

    # Filter points with an invalid state. - 0 is an invalid state
    valid = [p in invalid_states for p in points[-4, :]]
    points = points[:, valid]

    # Filter by dynProp. - 0 - 6 are moving, 7 is stopped
    valid = [p in dynprop_states for p in points[3, :]]
    points = points[:, valid]

    # Filter by ambig_state. - Make sure Doppler ambiguity is unambiguous
    valid = [p in ambig_states for p in points[11, :]]
    points = points[:, valid]

    return points, fields

#Helper function to work with radar through the sweeps
#Returns a list of all radars (in 5 sublists) of the scene
#Input is the first sample of a scene
def get_scene_radars_sweeps(first_sample):
    #Create a list that contains 5 lists with each radars data through the scene
    #radar_front = nusc.get('sample_data', first_sample['data']['RADAR_FRONT'])
    channels = ['RADAR_FRONT', 'RADAR_FRONT_LEFT', 'RADAR_FRONT_RIGHT', 'RADAR_BACK_LEFT', 'RADAR_BACK_RIGHT']
    radars = []
    for channel in channels:
        one_radar = nusc.get('sample_data', first_sample['data'][channel])
        one_radar_scene = []
        while one_radar['next'] != '':
            one_radar_scene.append(one_radar)
            one_radar = nusc.get('sample_data', one_radar['next'])
        radars.append(one_radar_scene)
    # #Create list of 5 timestamp arrrays
    # radars_time = [np.asarray([rad['timestamp'] for rad in rads]) for rads in radars]
    return radars

def interactive_vis_radar(scene, record=False):
    first_sample = nusc.get('sample', scene['first_sample_token'])
    last_sample = nusc.get('sample', scene['last_sample_token'])
    sample = first_sample
    pcd = o3d.geometry.PointCloud()
    
    if record:
        name = scene['name'] + "_interactive_radar.avi"
        out = None
 
    #Render callback for non blocking visualization which allows smoother control of view
    #Define the callback here to simplify the transfer of variables samples, and pcd
    def render_callback(vis):
        nonlocal sample, name, out
        if sample['token'] == last_sample['token']:
            return False

        #Load next scan
        #radar = nusc.get('sample_data', sample['data']['RADAR_FRONT'])
        #radar_file = nusc.get_sample_data_path(radar['token'])
        #scan, fields = radar_from_file(radar_file) #scan 18 x N
        #########################################
        #Replace above code with this to fuse all radar data
        scan, fields = fuse_radars_in_ego(sample) #18xN
        
        #points = np.rint(scan).astype(int).transpose() #change to N x 18 for o3d
        points = np.rint(scan).astype(int).transpose()[:, :3] #transpose/slice to N x 3 for o3d

        #Colorise distance using jet colormap
        #color = np.copy(points)
        #Use l2 norm to get distance of point
        dist = np.sqrt(np.sum(np.square(points), axis=1))
    
        #Normalize distance
        dist = ((dist - np.min(dist)) / (np.max(dist) - np.min(dist)))

        # color_map = cv2.COLORMAP_JET
        # color = cv2.applyColorMap(np.uint8(dist*255), color_map).squeeze()/255 #Opencv color range [0, 255]
        # #Use advanced slicing to change the colors from BGR to RGB
        # color[:, [2, 1, 0]] = color[:, [0, 1, 2]]
        #Use pyplot colormap - I like viridis more than jet
        color = plt.get_cmap('viridis')(dist)[:, :3] #matplotlib color range [0, 1]

        #Update Geometry
        if sample['token'] == first_sample['token']:
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.colors = o3d.utility.Vector3dVector(color)
            vis.add_geometry(pcd)
        pcd.points = o3d.utility.Vector3dVector(points) #Only need XYZ
        pcd.colors = o3d.utility.Vector3dVector(color)
        #vis.clear_geometries()
        #vis.add_geometry(pcd)

        # Update the visualization window and process events to handle keyboard inputs.
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
        time.sleep(0.4)

        return True

    
    # Create an Open3D visualization window.
    vis = o3d.visualization.VisualizerWithKeyCallback()

    # Set the custom update function.
    vis.register_animation_callback(render_callback)
    

    # Start animation.
    vis.create_window()

    opt = vis.get_render_option()
    opt.background_color = np.asarray([0, 0, 0])
    opt.point_size = 5 #Original 5

    vis.run()
    vis.destroy_window()
    if record:
        out.release()

#Visualized radar point cloud from the sweeps
def interactive_vis_radar_sweeps(scene, record=False):
    first_sample = nusc.get('sample', scene['first_sample_token'])
    last_sample = nusc.get('sample', scene['last_sample_token'])
    sample = first_sample
    pcd = o3d.geometry.PointCloud()

    first = True

    if record:
        name = scene['name'] + "_interactive_radar.avi"
        out = None
    radar_data =np.empty((18, 0))

    #Create a list that contains 5 lists with each radars data through the scene
    radar_front = nusc.get('sample_data', first_sample['data']['RADAR_FRONT'])
    channels = ['RADAR_FRONT', 'RADAR_FRONT_LEFT', 'RADAR_FRONT_RIGHT', 'RADAR_BACK_LEFT', 'RADAR_BACK_RIGHT']
    radars = []
    for channel in channels:
        one_radar = nusc.get('sample_data', first_sample['data'][channel])
        one_radar_scene = []
        while one_radar['next'] != '':
            one_radar_scene.append(one_radar)
            one_radar = nusc.get('sample_data', one_radar['next'])
        radars.append(one_radar_scene)
    #Create list of 5 timestamp arrrays
    radars_time = [np.asarray([rad['timestamp'] for rad in rads]) for rads in radars]

    #Render callback for non blocking visualization which allows smoother control of view
    #Define the callback here to simplify the transfer of variables samples, and pcd
    def render_callback(vis):
        nonlocal radar_front, name, out, first, radars_time

        #I have radar_front, now I need to get the corresponding other radars
        diffs = [np.asarray(rad_time - radar_front['timestamp']) for rad_time in radars_time]
        #Ensure that other radars are older than radar_front - This is not neccessary
        diffs = [np.where(dif > 0, dif, np.inf) for dif in diffs]
        diffs = np.abs(diffs)
            
        radar_front_left = radars[1][diffs[1].argmin()]
        radar_front_right = radars[2][diffs[2].argmin()]
        radar_back_left = radars[3][diffs[3].argmin()]
        radar_back_right = radars[4][diffs[4].argmin()]
        
        #This frames list of radars which lie temporally close
        radar_set = [radar_front, radar_front_left, radar_front_right, radar_back_left, radar_back_right]
        scan, fields = fuse_radars_in_ego_sweeps(radar_set) #18xN
        
        #points = np.rint(scan).astype(int).transpose() #change to N x 18 for o3d
        points = np.rint(scan).astype(int).transpose()[:, :3] #transpose/slice to N x 3 for o3d

        #Use l2 norm to get distance of point
        dist = np.sqrt(np.sum(np.square(points), axis=1))
        #Normalize distance
        dist = ((dist - np.min(dist)) / (np.max(dist) - np.min(dist)))
        color = plt.get_cmap('viridis')(dist)[:, :3] #matplotlib color range [0, 1]

        #Update Geometry
        if first:
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.colors = o3d.utility.Vector3dVector(color)
            vis.add_geometry(pcd)
            first = False
        pcd.points = o3d.utility.Vector3dVector(points) #Only need XYZ
        pcd.colors = o3d.utility.Vector3dVector(color)

        # Update the visualization window and process events to handle keyboard inputs.
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
        if radar_front['next'] == '':
            return False
        radar_front = nusc.get('sample_data', radar_front['next'])

        #Slow down animation
        time.sleep(0.1)
        return True

    # Create an Open3D visualization window.
    vis = o3d.visualization.VisualizerWithKeyCallback()
    # Set the custom update function.
    vis.register_animation_callback(render_callback)
    # Start animation.
    vis.create_window()

    opt = vis.get_render_option()
    opt.background_color = np.asarray([0, 0, 0])
    opt.point_size = 5 #Original 5

    vis.run()
    vis.destroy_window()
    if record:
        out.release()

#Use Open3D to visualize a fixed top down view on lidar throughout scene
def top_down_radar(scene, record=False):
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
    opt.point_size = 4 #Original 5

    if record:
        name = scene['name'] + "_top_down_radar.avi"
        out = None

    while sample['token'] != last_sample['token']:
         #Load next scan
        # radar = nusc.get('sample_data', sample['data']['RADAR_FRONT'])
        # radar_file = nusc.get_sample_data_path(radar['token'])
        # scan, fields = radar_from_file(radar_file) #scan 18 x N
        #points = np.rint(scan).astype(int).transpose() #change to N x 18 for o3d

        scan, fields = fuse_radars_in_ego(sample)
        points = np.rint(scan).astype(int).transpose()[:, :3]

        #Use l2 norm to get distance of point
        dist = np.sqrt(np.sum(np.square(points), axis=1))
        #Normalize distance
        dist = ((dist - np.min(dist)) / (np.max(dist) - np.min(dist)))
        color = plt.get_cmap('viridis')(dist)[:, :3] #matplotlib color range [0, 1]

        #Update Geometry
        pcd.points = o3d.utility.Vector3dVector(points) #Only need XYZ
        pcd.colors = o3d.utility.Vector3dVector(color)
        vis.clear_geometries()
        vis.add_geometry(pcd)

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
        time.sleep(0.25)

    vis.destroy_window()
    if record:
        out.release()
    

#Returns np array of all radar points in ego frame
def fuse_radars_in_ego(sample):
    channels = ['RADAR_FRONT', 'RADAR_FRONT_LEFT', 'RADAR_FRONT_RIGHT', 'RADAR_BACK_LEFT', 'RADAR_BACK_RIGHT']
    #Merged points in ego frame
    radar_data =np.empty((18, 0))
    for channel in channels:
        #Load Radar in its own frame
        radar = nusc.get('sample_data', sample['data'][channel])
        radar_file = nusc.get_sample_data_path(radar['token'])
        scan, fields = radar_from_file(radar_file) #scan 18 x N
        # Transform radar points into coordinate frame of ego
        #Get Lidar Pose
        radar_sensor = nusc.get('calibrated_sensor', radar['calibrated_sensor_token'])
        radar_t = np.asarray(radar_sensor['translation'])
        radar_r_quat = Quaternion(radar_sensor['rotation'])

        #Rotate and translate
        scan[:3, :] = np.dot(radar_r_quat.rotation_matrix, scan[:3, :])
        scan[:3, :] = (scan[:3, :].transpose() + radar_t).transpose()

        radar_data = np.append(radar_data, scan, axis=1)
    return radar_data, fields

#Returns np array of all radar points in ego frame
#Input is a list of the 5 radars which should be merged
def fuse_radars_in_ego_sweeps(radars):
    #Merged points in ego frame
    radar_data =np.empty((18, 0))
    for radar in radars:
        #Load Radar in its own frame
        radar_file = nusc.get_sample_data_path(radar['token'])
        scan, fields = radar_from_file(radar_file) #scan 18 x N
        # Transform radar points into coordinate frame of ego
        #Get Lidar Pose
        radar_sensor = nusc.get('calibrated_sensor', radar['calibrated_sensor_token'])
        radar_t = np.asarray(radar_sensor['translation'])
        radar_r_quat = Quaternion(radar_sensor['rotation'])

        #Rotate and translate
        scan[:3, :] = np.dot(radar_r_quat.rotation_matrix, scan[:3, :])
        scan[:3, :] = (scan[:3, :].transpose() + radar_t).transpose()

        radar_data = np.append(radar_data, scan, axis=1)
    return radar_data, fields

#Show video of selected camera with radar rendered into the image
def visualize_radar_in_img(scene, cam_channel='CAM_FRONT', record=False):
    first_sample = nusc.get('sample', scene['first_sample_token'])
    last_sample = nusc.get('sample', scene['last_sample_token'])

    winname = "Radar - " + cam_channel
    window_width = nusc.get('sample_data', first_sample['data'][cam_channel])['width']
    window_height = nusc.get('sample_data', first_sample['data'][cam_channel])['height']
    cv2.namedWindow(winname, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(winname, window_width, window_height)

    if record:
        name = scene['name'] + "_radar_in_img_" + cam_channel + ".avi"
        out = cv2.VideoWriter(name, cv2.VideoWriter_fourcc(*'DIVX'), 2, (window_width, window_height))

    #Load lidar and img data from scan
    sample = first_sample

    while sample['token'] != last_sample['token']:
        if sample['token'] == last_sample['token']:
            sample = first_sample
        #Load camera image
        cam = nusc.get('sample_data', sample['data'][cam_channel])
        cam_file = os.path.join(nusc.dataroot, cam['filename'])
        cam_img = cv2.imread(cam_file)

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

        # #First transformation - Skip because Radar points are alread in ego frame
        # pcd.rotate(radar_r_quat.rotation_matrix, np.array([0, 0, 0]))
        # pcd.translate(radar_t)

        #Second transformation - From ego to camera frame
        pcd.translate(-cam_t)
        pcd.rotate(cam_r_quat.rotation_matrix.T, np.array([0, 0, 0]))

        #Project points into image plane
        view = np.copy(cam_k)
        points = np.asarray(pcd.points).T[:3, :]

        #Set color to depth value
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
        points = points[:, mask]

        color_img = np.copy(depth)
        color_img = color_img[mask]

        color = np.zeros(points.shape).transpose()
        scan = scan[:, mask]
        for i in range(color.shape[0]):
            color[i] = [255, 0, 0] if scan[8, i] > 0 else [0, 0, 255]

        #Draw Radar points in image - Red for moving towards ego, blue for moving away from ego
        for i in range(points.shape[1]):
            p_loc = (int(np.rint(points[0, i])), int(np.rint(points[1, i])))
            #text = "vX: " + "{:.5f}".format(scan[8, i]) + ", vY: " + "{:.5f}".format(scan[9, i])
            cv2.circle(cam_img, p_loc, 4, color[i], -1)
            #cv2.putText(cam_img, text, p_loc, cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),thickness=1)
        # i = int(points.shape[1] / 4)
        # p_loc = (int(np.rint(points[0, i])), int(np.rint(points[1, i])))
        # text = "vX: " + "{:.5f}".format(scan[8, i]) + ", vY: " + "{:.5f}".format(scan[9, i])
        # cv2.circle(cam_img, p_loc, 4, color[i], -1)
        # cv2.putText(cam_img, text, p_loc, cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),thickness=1)

        if record:
            out.write(cam_img)

        cv2.imshow(winname, cam_img)
        key = cv2.waitKey(400)
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

#Show video of selected camera with radar rendered into the image - Over all sweeps
def visualize_radar_in_img_sweeps(scene, cam_channel='CAM_FRONT', record=False):
    first_sample = nusc.get('sample', scene['first_sample_token'])
    last_sample = nusc.get('sample', scene['last_sample_token'])

    winname = "Radar - " + cam_channel
    window_width = nusc.get('sample_data', first_sample['data'][cam_channel])['width']
    window_height = nusc.get('sample_data', first_sample['data'][cam_channel])['height']
    cv2.namedWindow(winname, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(winname, window_width, window_height)

    if record:
        name = scene['name'] + "_radar_in_img_" + cam_channel + ".avi"
        out = cv2.VideoWriter(name, cv2.VideoWriter_fourcc(*'DIVX'), 2, (window_width, window_height))

    # #Create a list that contains 5 lists with each radars data through the scene
    radar_front = nusc.get('sample_data', first_sample['data']['RADAR_FRONT'])
    # channels = ['RADAR_FRONT', 'RADAR_FRONT_LEFT', 'RADAR_FRONT_RIGHT', 'RADAR_BACK_LEFT', 'RADAR_BACK_RIGHT']
    # radars = []
    # for channel in channels:
    #     one_radar = nusc.get('sample_data', first_sample['data'][channel])
    #     one_radar_scene = []
    #     while one_radar['next'] != '':
    #         one_radar_scene.append(one_radar)
    #         one_radar = nusc.get('sample_data', one_radar['next'])
    #     radars.append(one_radar_scene)
    radars = get_scene_radars_sweeps(first_sample)
    #Create list of 5 timestamp arrrays
    radars_time = [np.asarray([rad['timestamp'] for rad in rads]) for rads in radars]

    cam = nusc.get('sample_data', first_sample['data'][cam_channel])
    while True:
        #Load camera image
        cam_file = os.path.join(nusc.dataroot, cam['filename'])
        cam_img = cv2.imread(cam_file)

        #I have radar_front, now I need to get the corresponding other radars
        diffs = [np.asarray(rad_time - radar_front['timestamp']) for rad_time in radars_time]
        #Ensure that other radars are older than radar_front - This is not neccessary
        diffs = [np.where(dif > 0, dif, np.inf) for dif in diffs]
        diffs = np.abs(diffs)

        #  
        radar_front_left = radars[1][diffs[1].argmin()]
        radar_front_right = radars[2][diffs[2].argmin()]
        radar_back_left = radars[3][diffs[3].argmin()]
        radar_back_right = radars[4][diffs[4].argmin()]
        
        #This frames list of radars which lie temporally close
        radar_set = [radar_front, radar_front_left, radar_front_right, radar_back_left, radar_back_right]
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

        # #First transformation - Skip because Radar points are alread in ego frame
        # pcd.rotate(radar_r_quat.rotation_matrix, np.array([0, 0, 0]))
        # pcd.translate(radar_t)

        #Second transformation - From ego to camera frame
        pcd.translate(-cam_t)
        pcd.rotate(cam_r_quat.rotation_matrix.T, np.array([0, 0, 0]))

        #Project points into image plane
        view = np.copy(cam_k)
        points = np.asarray(pcd.points).T[:3, :]

        #Set color to depth value
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
        points = points[:, mask]

        color_img = np.copy(depth)
        color_img = color_img[mask]

        color = np.zeros(points.shape).transpose()
        scan = scan[:, mask]
        for i in range(color.shape[0]):
            color[i] = [255, 0, 0] if scan[8, i] > 0 else [0, 0, 255]

        #Draw Radar points in image - Red for moving towards ego, blue for moving away from ego
        for i in range(points.shape[1]):
            p_loc = (int(np.rint(points[0, i])), int(np.rint(points[1, i])))
            #text = "vX: " + "{:.5f}".format(scan[8, i]) + ", vY: " + "{:.5f}".format(scan[9, i])
            cv2.circle(cam_img, p_loc, 4, color[i], -1)
            #cv2.putText(cam_img, text, p_loc, cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),thickness=1)
        # i = int(points.shape[1] / 4)
        # p_loc = (int(np.rint(points[0, i])), int(np.rint(points[1, i])))
        # text = "vX: " + "{:.5f}".format(scan[8, i]) + ", vY: " + "{:.5f}".format(scan[9, i])
        # cv2.circle(cam_img, p_loc, 4, color[i], -1)
        # cv2.putText(cam_img, text, p_loc, cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),thickness=1)

        if record:
            out.write(cam_img)

        cv2.imshow(winname, cam_img)
        key = cv2.waitKey(50)
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
        radar_diff = radars_time[0] - cam['timestamp']
        radar_diff = np.where(radar_diff > 0, radar_diff, np.inf)
        radar_front = radars[0][radar_diff.argmin()]
    cv2.destroyAllWindows()
    if record:
        out.release()

##################################################
##################################################
