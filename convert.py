import numpy as np
from PIL import Image, ImageDraw
import open3d as o3d

def pinhole_projection(point_3d):
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
    focal_length = 10
    sensor_width = 36
    sensor_height = 24
    camera_distance = 500
    
    # Project the 3D point onto the image plane
    projection_x = (focal_length * point_3d[0]) / (point_3d[2] + camera_distance)
    projection_y = (focal_length * point_3d[1]) / (point_3d[2] + camera_distance)
    projection_h = np.sqrt(point_3d[0] ** 2 + point_3d[1] ** 2)
    
    # Convert to pixel coordinates
    pixel_x = (projection_x + sensor_width / 2)
    pixel_y = (sensor_height / 2 - projection_y)
    pixel_depth = np.sqrt(projection_h ** 2 + point_3d[2] ** 2)
    
    return np.array([pixel_x, pixel_y, pixel_depth]) 

def main():
    # Define the 3D point and camera parameters
    point_load = o3d.io.read_point_cloud("./longdress.ply")
    point_array = np.asarray(point_load.points)   # Example 3D point
    
    # Rasterize the points and save them to a list
    x_list = []
    y_list = []
    depth_list = []

    # Create progress markers
    quarter_length = int(len(point_array)/4)
    half_length = int(len(point_array)/2)
    three_quarter_length = int(3*len(point_array)/4)

    for index, world_point in enumerate(point_array):
        raster_point = pinhole_projection(world_point)
        x_list.append(raster_point[0])
        y_list.append(raster_point[1])
        depth_list.append(raster_point[2])
        if (index == quarter_length or index == half_length or index == three_quarter_length):
            print("point " , index , " successfully processed.")

    img_width = 1024
    img_height = 1024
    scale_factor = 40
    color_factor = 0.3
    x_offset = -400
    y_offset = 300

    depth_matrix = np.zeros((img_height, img_width, 3), np.uint8) # 1 because grayscale

    for i in range(len(depth_list)):
        x_position = x_list[i] * scale_factor + x_offset
        y_position = y_list[i] * scale_factor + y_offset
        color = np.clip(depth_list[i] * color_factor, 0, 255)
        if ((x_position < img_width and x_position > 0) and (y_position < img_height and y_position > 0)):
            if depth_matrix[int(y_position), int(x_position), 0] < color:
                depth_matrix[int(y_position), int(x_position)] = [color, color, color]


    # Draw them as an image
    img = Image.fromarray(depth_matrix)
    img.show()

if __name__ == "__main__":
    main()