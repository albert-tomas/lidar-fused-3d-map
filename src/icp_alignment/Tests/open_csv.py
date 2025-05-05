import os
import pandas as pd
import open3d as o3d
import numpy as np
import time
import pygetwindow as gw

def get_csv_files(folder_path):
    # Returns a list of CSV files in the given directory
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    # Check if there are any CSV files 
    if not csv_files:
        print("No CSV files found in the folder.")
    else:
        print(f"Found {len(csv_files)} CSV files.")
    return csv_files

def load_point_clouds(csv_files, folder_path, max_number_of_clouds=10):
    # Loads up to max_iterations point clouds from CSV files
    # Each point cloud has the x, y, z coordinates and the file name
    point_clouds = []
    for i, file in enumerate(csv_files):
        if i >= max_number_of_clouds:
            break
        if i >= 83:
            full_path = os.path.join(folder_path, file)
            pc = pd.read_csv(full_path, usecols=["x(m)", "y(m)", "z(m)"])
            pc["file"] = file
            point_clouds.append(pc)
            print(i)
    return point_clouds

def create_open3d_point_cloud(df):
    # Converts a DataFrame into an Open3D PointCloud object
    x, y, z = df["x(m)"], df["y(m)"], df["z(m)"]
    points = np.vstack((x, y, z)).T
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(points)
    # TODO: maybe return the points as well if they have to be processed after, avoiding recalculating
    return pc

def setup_visualizer():
    # Creates and configures an Open3D visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    # Get the window and maximize it
    window = gw.getWindowsWithTitle('Open3D')[0]
    window.maximize()
    return vis

def display_point_cloud(vis, point_cloud, point_size=2.0):
    # Displays the point cloud in the visualizer
    vis.clear_geometries()
    vis.add_geometry(point_cloud)
    render_option = vis.get_render_option()
    render_option.point_size = point_size
    
    #time.sleep(1)
    vis.update_geometry(point_cloud)
    vis.poll_events()
    vis.update_renderer()
    #vis.run()    

if __name__ == "__main__":
    csv_files = get_csv_files("D:/lidar-thesis/CSV_examples/renamed")
    pc = load_point_clouds(csv_files, "D:/lidar-thesis/CSV_examples/renamed", 84)
    vis = setup_visualizer()
    
    display_point_cloud(vis, create_open3d_point_cloud(pc[0]), 2.0)
    vis.run()