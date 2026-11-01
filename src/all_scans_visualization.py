import os
import numpy as np
import open3d as o3d
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm


print("--- Initializing Configuration ---")

CSV_SCAN_DIRECTORY = '/home/etss/Desktop/SLAM/10-06-25/Combined_RVP_csv/' 
ESTIMATED_TRAJ_PATH = '/home/etss/Desktop/SLAM/12_06/estimated_trajectory_icp.txt'
VOXEL_SIZE = 1.0

MAX_PLOT_POINTS = 750000

ANIMATION_FPS = 30

SAVE_VIDEO = True
OUTPUT_DIR = "slam_animation_output"
VIDEO_FILENAME = os.path.join(OUTPUT_DIR, "full_slam_visualization.gif")
print("-" * 60 + "\n")

def read_tum_trajectory(filepath):
    try:
        data = np.loadtxt(filepath, dtype=float)
        translations = data[:, 1:4]
        quaternions = data[:, 4:8]
        poses = []
        for t, q in zip(translations, quaternions):
            tm = np.identity(4)
            tm[:3, :3] = R.from_quat(q).as_matrix()
            tm[:3, 3] = t
            poses.append(tm)
        return poses, translations
    except Exception as e:
        print(f"ERROR: Could not read trajectory file '{filepath}': {e}")
        return None, None

def read_lidar_scan_from_csv(filepath):
    try:
        points = np.loadtxt(filepath, delimiter=',', usecols=(0, 1, 2), skiprows=1)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        return pcd
    except Exception as e:
        if 'is empty' in str(e):
            return o3d.geometry.PointCloud()
        print(f"ERROR: Could not read CSV file '{filepath}': {e}")
        return None

def main():

    print("--- Loading all available data ---")
    est_poses, est_translations = read_tum_trajectory(ESTIMATED_TRAJ_PATH)
    all_csv_files = sorted([os.path.join(CSV_SCAN_DIRECTORY, f) for f in os.listdir(CSV_SCAN_DIRECTORY) if f.endswith('.csv')])

    num_poses = len(est_poses) if est_poses else 0
    num_scans = len(all_csv_files)
    num_frames = min(num_poses, num_scans)

    if num_frames == 0:
        print("CRITICAL ERROR: Could not find any scans or poses. Check your paths in the configuration. Aborting.")
        return
        
    print(f"\n--- Data Counts ---")
    print(f"Found {num_poses} estimated poses.")
    print(f"Found {num_scans} CSV scan files.")
    print(f"--> Running visualization for {num_frames} frames (the smallest available count).")
    
    
    est_poses = est_poses[:num_frames]
    est_translations = est_translations[:num_frames]
    csv_files = all_csv_files[:num_frames]


    print(f"\n--- Aggregating all {num_frames} LiDAR scans. This will take some time... ---")
    all_points_list = []
    for i in tqdm(range(num_frames), desc="Processing Scans"):
        pcd = read_lidar_scan_from_csv(csv_files[i])
        if pcd and not pcd.is_empty():
            pcd = pcd.voxel_down_sample(voxel_size=VOXEL_SIZE)
            pcd.transform(est_poses[i])
            all_points_list.append(np.asarray(pcd.points))

    full_point_cloud = np.concatenate(all_points_list, axis=0)
    print(f"--- Full point cloud map created with {len(full_point_cloud)} points. ---")


    print("\n--- Preparing animation plot ---")
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.set_facecolor('black')
    
    if len(full_point_cloud) > MAX_PLOT_POINTS:
        print(f"--- Subsampling background points to {MAX_PLOT_POINTS} for performance ---")
        idx = np.random.choice(len(full_point_cloud), size=MAX_PLOT_POINTS, replace=False)
        plot_points = full_point_cloud[idx]
    else:
        plot_points = full_point_cloud
    
    ax.scatter(plot_points[:, 0], plot_points[:, 1], s=0.1, c='grey', alpha=0.5)

    est_line, = ax.plot([], [], 'r-', linewidth=2, label='Estimated Trajectory')
    robot_marker, = ax.plot([], [], 'go', markersize=10, label='Robot Position')

    ax.legend()
    ax.set_title("SLAM Visualization - Frame 0")
    ax.set_xlabel("X (meters)")
    ax.set_ylabel("Y (meters)")
    ax.axis('equal')

    def animate(i):
        est_line.set_data(est_translations[:i+1, 0], est_translations[:i+1, 1])
        current_pose = est_poses[i]
        robot_marker.set_data([current_pose[0, 3]], [current_pose[1, 3]])
        ax.set_title(f"SLAM Visualization - Frame {i+1}/{num_frames}")
        return [est_line, robot_marker]

    anim = FuncAnimation(
        fig,
        animate,
        frames=num_frames,
        blit=False, 
        interval=1000/ANIMATION_FPS,
        repeat=False
    )

    if SAVE_VIDEO:
        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)
        print(f"\n--- Saving animation to '{VIDEO_FILENAME}' (this will be slow for many frames)... ---")
        pbar = tqdm(total=num_frames, desc="Saving Video")
        anim.save(VIDEO_FILENAME, writer='pillow', fps=ANIMATION_FPS, dpi=150, progress_callback=lambda i, n: pbar.update(1))
        pbar.close()
        print(f"--- Animation saved successfully to '{VIDEO_FILENAME}'. ---")
    else:
        matplotlib.use('TkAgg')
        print("\n--- Displaying live animation window. Close the window to exit. ---")
        plt.show()

if __name__ == '__main__':
    main()
