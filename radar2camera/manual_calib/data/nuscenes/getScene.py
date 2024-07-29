import os
import struct

import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
import open3d as o3d
import shutil
import pandas as pd

# nuScenes veri setinin yüklü olduğu yolu belirtin.
dataroot = '/home/mahsun/nuscenes/dataset/v1.0-mini'
nusc = NuScenes(version='v1.0-mini', dataroot=dataroot, verbose=True)

# Örnek bir senaryo seçin.
sample = nusc.sample[10]

# Ön kameraya ve ön radara ait örnek veri noktalarını (sample data) bulun.
front_camera_data = None
front_radar_data = None
lidar_top_data = None

for sd in sample['data']:
    if 'CAM_FRONT' in sd:
        front_camera_data = nusc.get('sample_data', sample['data']['CAM_FRONT'])
        # Get the calibrated sensor data for the front camera
        front_camera_calib = nusc.get('calibrated_sensor', front_camera_data['calibrated_sensor_token'])
    elif 'RADAR_FRONT' in sd:
        front_radar_data = nusc.get('sample_data', sample['data']['RADAR_FRONT'])
    elif 'LIDAR_TOP' in sd:
        lidar_top_data = nusc.get('sample_data', sample['data']['LIDAR_TOP'])


# Ön kamera verisini yükleyin.
if front_camera_data is not None:
    front_camera_intrinsics = front_camera_calib['camera_intrinsic']
    print("---------- Front Camera Intrinsic ----------")
    print(f"Front Camera Intrinsic: {front_camera_intrinsics}")

    front_camera_filepath = nusc.get_sample_data_path(front_camera_data['token'])
    print("---------- Front Camera Filepath  ----------")
    print(f"Front Camera Filepath: {front_camera_filepath}")

# Ön radar verisini yükleyin.
if front_radar_data is not None:
    front_radar_filepath = nusc.get_sample_data_path(front_radar_data['token'])
    print("---------- Front Radar Filepath  ----------")
    print(f"Front Radar Filepath: {front_radar_filepath}")

# Lidar verisini yükleyin.
if lidar_top_data is not None:
    lidar_top_filepath = nusc.get_sample_data_path(lidar_top_data['token'])
    print("---------- Lidar TopFilepath  ----------")
    print(f"Lidar Top Filepath: {lidar_top_filepath}")


# Load the .pcd.bin file.
pc = LidarPointCloud.from_file(lidar_top_filepath)
bin_pcd = pc.points.T

# Reshape and get only values for x, y and z.
bin_pcd = bin_pcd.reshape((-1, 4))[:, 0:3]

# Convert to Open3D point cloud.
o3d_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(bin_pcd))

# Save to a .pcd file.
o3d.io.write_point_cloud(os.path.expanduser("/home/mahsun/masterThesisStudies/SensorsCalibration/lidar2camera/manual_calib/data/nuscenes/0.pcd"), o3d_pcd)

# Read the saved .pcd file from the previous step.
pcd = o3d.io.read_point_cloud(os.path.expanduser("/home/mahsun/masterThesisStudies/SensorsCalibration/lidar2camera/manual_calib/data/nuscenes/0.pcd"))
out_arr = np.asarray(pcd.points)  

# Load the original point cloud data from nuScenes, and check that the saved .pcd matches the original data.
pc = LidarPointCloud.from_file(lidar_top_filepath)
points = pc.points.T
assert np.array_equal(out_arr, points[:, :3])

shutil.copy(front_camera_filepath, "/home/mahsun/masterThesisStudies/SensorsCalibration/lidar2camera/manual_calib/data/nuscenes/0.png")
shutil.copy(front_camera_filepath, "/home/mahsun/masterThesisStudies/SensorsCalibration/radar2camera/manual_calib/data/nuscenes/0.png")
shutil.copy(front_radar_filepath, "/home/mahsun/masterThesisStudies/SensorsCalibration/radar2camera/manual_calib/data/nuscenes/0.pcd")


# ----------------- Convert pcd to csv -----------------

# Load the point cloud
pcd = o3d.io.read_point_cloud("/home/mahsun/masterThesisStudies/SensorsCalibration/radar2camera/manual_calib/data/nuscenes/0.pcd")
points = np.asarray(pcd.points)

# Convert to DataFrame
df = pd.DataFrame(points, columns=["x", "y" , "z"])

# Save to CSV
df.to_csv("/home/mahsun/masterThesisStudies/SensorsCalibration/radar2camera/manual_calib/data/nuscenes/output.csv", index=False)
