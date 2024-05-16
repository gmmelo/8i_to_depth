import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

def pinhole_projection(point_3d, camera_params):
    """
    Project a 3D point onto a 2D image plane using the pinhole camera model.
    
    Args:
    - point_3d: A numpy array representing the 3D point in Cartesian coordinates (x, y, z).
    - camera_params: A dictionary containing camera parameters:
        - 'focal_length': Focal length of the camera.
        - 'sensor_width': Width of the image sensor.
        - 'sensor_height': Height of the image sensor.
        
    Returns:
    - A numpy array representing the 2D projection of the 3D point on the image plane.
    """
    focal_length = camera_params['focal_length']
    sensor_width = camera_params['sensor_width']
    sensor_height = camera_params['sensor_height']
    
    # Project the 3D point onto the image plane
    projection_x = (focal_length * point_3d[0]) / point_3d[2]
    projection_y = (focal_length * point_3d[1]) / point_3d[2]
    
    # Convert to pixel coordinates
    pixel_x = (projection_x + sensor_width / 2)
    pixel_y = (sensor_height / 2 - projection_y)
    
    return np.array([pixel_x, pixel_y])

def plot_projection(point_3d, point_2d):
    """
    Plot the 3D point and its projection on a 2D image plane.
    
    Args:
    - point_3d: A numpy array representing the 3D point in Cartesian coordinates (x, y, z).
    - point_2d: A numpy array representing the 2D projection of the 3D point on the image plane.
    """
    plt.figure()
    
    # Plot the 3D point
    plt.plot(point_3d[0], point_3d[1], 'bo', label='3D Point')
    
    # Plot the projection
    plt.plot(point_2d[0], point_2d[1], 'ro', label='2D Projection')
    
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.gca().invert_yaxis()  # Invert y-axis to match image coordinates
    plt.legend()
    

def main():
    # Define the 3D point and camera parameters
    point_load = o3d.io.read_point_cloud("./longdress_vox10_1051.ply")
    point_array = np.asarray(point_load.points)   # Example 3D point

    camera_params = {
        'focal_length': 10,     # Focal length in mm
        'sensor_width': 36,     # Width of the image sensor in mm (e.g., 35mm film)
        'sensor_height': 24,    # Height of the image sensor in mm (e.g., 35mm film)
    }
    
    plt.figure()

    for point in point_array:
        # For each point in the point cloud
        point_3d = point
        # Project the 3D point onto the 2D image plane
        point_2d = pinhole_projection(point_3d, camera_params)
        
        # Plot the 3D point
        plt.plot(point_3d[0], point_3d[1], 'bo', label='3D Point')
        
        # Plot the projection
        plt.plot(point_2d[0], point_2d[1], 'ro', label='2D Projection')

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.gca().invert_yaxis()  # Invert y-axis to match image coordinates
    plt.show()

if __name__ == "__main__":
    main()