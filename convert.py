import numpy as np
from PIL import Image, ImageDraw
import open3d as o3d

img_width_cm = 0.24
img_height_cm = 0.24
focal_length_cm = 0.1
img_width_pixels = 1024
img_height_pixels = 1024
physical_to_pixels = img_width_pixels / img_width_cm

def pinhole_projection(point_3d):
    """
    Project a 3D point onto a 2D image plane using the pinhole camera model.
    
    Arg:
    - point_3d: A numpy array representing the 3D point in Cartesian coordinates (x, y, z).
        
    Returns:
    - A numpy array representing the 2D projection of the 3D point on the image plane.
    """
    
    # Project the 3D point onto the image plane
    projection_x = (focal_length_cm * point_3d[0][0]) / (point_3d[2][0])
    projection_y = (focal_length_cm * point_3d[1][0]) / (point_3d[2][0])
    projection_h = np.sqrt(point_3d[0][0] ** 2 + point_3d[1][0] ** 2)
    
    # Convert to pixel coordinates
    pixel_x = (projection_x + img_width_cm / 2)
    pixel_y = (img_height_cm / 2 - projection_y)
    pixel_depth = np.sqrt(projection_h ** 2 + point_3d[2][0] ** 2)
    
    return np.array([pixel_x, pixel_y, pixel_depth]) 

def main():
    # Define the 3D point and camera parameters
    point_load = o3d.io.read_point_cloud("./longdress.ply")
    point_array = np.asarray(point_load.points)   # Example 3D point
    
    # Rasterize the points and save them to a list
    point_count = point_array.shape[0]
    x_list = np.empty(point_count)
    y_list = np.empty(point_count)
    depth_list = np.empty(point_count)
    
    # Create the extrinsic matrix
    camera_x = 0
    camera_y = -500
    camera_z = 400
    world_to_camera = np.array([[1, 0, 0, camera_x],
                                [0, 1, 0, camera_y],
                                [0, 0, 1, camera_z]])

    # Create progress markers
    quarter_length = int(len(point_array)/4)
    half_length = int(len(point_array)/2)
    three_quarter_length = int(3*len(point_array)/4)

    for index, world_point in enumerate(point_array):
        world_point_homogeneous = np.array([[world_point[0]],
                                            [world_point[1]],
                                            [world_point[2]],
                                            [1]])
        camera_point = world_to_camera @ world_point_homogeneous
        raster_point = pinhole_projection(camera_point)
        x_list[index] = raster_point[0]
        y_list[index] = raster_point[1]
        depth_list[index] = raster_point[2]
        if (index == quarter_length or index == half_length or index == three_quarter_length):
            print("point " , index , " successfully processed.")

    
    depth_matrix = np.zeros((img_height_pixels, img_width_pixels, 3), np.uint8)

    depth_max = max(depth_list)
    depth_min = min(depth_list)
    depth_delta = depth_max - depth_min

    for i in range(len(depth_list)):
        x_position = x_list[i] * physical_to_pixels
        y_position = y_list[i] * physical_to_pixels
        color = 255 * (1 - (depth_list[i] - depth_min) / (depth_delta)) # normalize, scale to 0-255, and invert
        if ((x_position < img_width_pixels and x_position >= 0) and (y_position < img_height_pixels and y_position >= 0)):
            if depth_matrix[int(y_position), int(x_position), 0] < color:
                depth_matrix[int(y_position), int(x_position)] = color


    # Draw them as an image
    img = Image.fromarray(depth_matrix)
    img.show()

if __name__ == "__main__":
    main()