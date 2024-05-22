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

def main():
    # Define the 3D point and camera parameters
    point_load = o3d.io.read_point_cloud("./longdress_vox10_1051.ply")
    point_array = np.asarray(point_load.points)   # Example 3D point

    camera_params = {
        'focal_length': 10,     # Focal length in mm
        'sensor_width': 36,     # Width of the image sensor in mm (e.g., 35mm film)
        'sensor_height': 24,    # Height of the image sensor in mm (e.g., 35mm film)
    }
    
    # Rasterize the points and save them to a list
    x_list = []
    y_list = []

    for index, world_point in enumerate(point_array):
        raster_point = pinhole_projection(world_point, camera_params)
        x_list.append(raster_point[0])
        y_list.append(raster_point[1])
        print("point " , index , " successfully processed.")

    # Draw them as an image
    plt.scatter(x_list, y_list)
    plt.show()

if __name__ == "__main__":
    main()