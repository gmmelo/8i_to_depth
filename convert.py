import numpy as np
from PIL import Image, ImageDraw
import open3d as o3d

camera_count = 4

img_width_cm = 0.24
img_height_cm = 0.24
focal_length_cm = 0.1
img_width_pixels = 1024
img_height_pixels = 1024
physical_to_pixels = img_width_pixels / img_width_cm

def extrinsic_matrix(camera_position_x, camera_position_y, camera_position_z, camera_rotation_x, camera_rotation_y, camera_rotation_z):
    # Rotation in euler angles
    camera_rotation_z *= np.pi / 180
    camera_rotation_y *= np.pi / 180
    camera_rotation_x *= np.pi / 180

    # Defining the matrix cells for rotation in order z, y, x
    n11 = np.cos(camera_rotation_y) * np.cos(camera_rotation_z)
    n12 = np.sin(camera_rotation_x) * np.sin(camera_rotation_y) * np.cos(camera_rotation_z) - np.cos(camera_rotation_x) * np.sin(camera_rotation_z)
    n13 = np.cos(camera_rotation_x) * np.sin(camera_rotation_y) * np.cos(camera_rotation_z) + np.sin(camera_rotation_x) * np.sin(camera_rotation_z)
    n21 = np.cos(camera_rotation_y) * np.sin(camera_rotation_z)
    n22 = np.sin(camera_rotation_x) * np.sin(camera_rotation_y) * np.sin(camera_rotation_z) + np.cos(camera_rotation_x) * np.cos(camera_rotation_z)
    n23 = np.cos(camera_rotation_x) * np.sin(camera_rotation_y) * np.sin(camera_rotation_z) - np.sin(camera_rotation_x) * np.cos(camera_rotation_z)
    n31 = -np.sin(camera_rotation_y)
    n32 = np.sin(camera_rotation_x) * np.cos(camera_rotation_y)
    n33 = np.cos(camera_rotation_x) * np.cos(camera_rotation_y)

    camera_extrinsic_matrix = np.array([[n11, n12, n13, camera_position_x],
                                         [n21, n22, n23, camera_position_y],
                                         [n31, n32, n33, camera_position_z],
                                         [  0,   0,   0,                 1]])

    return camera_extrinsic_matrix

def pinhole_projection(point_3d):
    """
    Project a 3D point onto a 2D image plane using the pinhole camera model.
    
    Arg:
    - point_3d: A numpy array representing the 3D point in Cartesian coordinates (x, y, z).
        
    Returns:
    - A numpy array representing the 2D projection of the 3D point on the image plane.
    """
    
    # Project the 3D point onto the image plane
    projection_x = -(focal_length_cm * point_3d[0][0]) / (point_3d[2][0]) # Invert signal because of projection properties
    projection_y = -(focal_length_cm * point_3d[1][0]) / (point_3d[2][0])
    projection_h = np.sqrt(point_3d[0][0] ** 2 + point_3d[1][0] ** 2)
    
    # Convert to pixel coordinates
    pixel_x = (projection_x + img_width_cm / 2)
    pixel_y = (projection_y + img_height_cm / 2)
    pixel_depth = np.sqrt(projection_h ** 2 + point_3d[2][0] ** 2)
    
    return np.array([pixel_x, pixel_y, pixel_depth]) 

def main():
    # Define the 3D point and camera parameters
    point_load = o3d.io.read_point_cloud("./longdress.ply")
    point_count = len(point_load.points)
    point_array = np.empty((camera_count, point_count, 3), dtype = np.float64)
    screen_point_array = np.empty((camera_count, 3, point_count)) # For each camera, store x, y, and distance of each point in screen cm coordinates
    camera_extrinsic_matrix = np.empty((camera_count, 4, 4), dtype = np.float64)

    for i in range(camera_count):
        point_array[i] = np.asarray(point_load.points)
        camera_extrinsic_matrix[i] = extrinsic_matrix(0, -600, 500, 0, (360/(i+1)) % 360, 0) # Get a 360 view from <camera_count> cameras
        print(f"[DEBUG] Camera[{i}] extrinsic matrix:\n", camera_extrinsic_matrix[i])

    # Create progress markers
    quarter_length = int(len(point_array[0])/4)
    half_length = int(len(point_array[0])/2)
    three_quarter_length = int(3*len(point_array[0])/4)

    # Loop through point cloud and add rasterized points to a list
    for camera_index in range(camera_count):
        for point_index, world_point in enumerate(point_array[camera_index]):
            world_point_homogeneous = np.array([[world_point[0]],
                                                [world_point[1]],
                                                [world_point[2]],
                                                [1]])
            camera_point = camera_extrinsic_matrix[camera_index] @ world_point_homogeneous
            raster_point = pinhole_projection(camera_point)
            screen_point_array[camera_index][0][point_index] = raster_point[0] # Screen X (cm)
            screen_point_array[camera_index][1][point_index] = raster_point[1] # Screen Y (cm)
            screen_point_array[camera_index][2][point_index] = raster_point[2] # Point distance (cm)
            if (point_index == quarter_length or point_index == half_length or point_index == three_quarter_length):
                print(f"point {point_index} successfully processed for camera {camera_index}")

    # Image matrix is for color data, depth matrix is form raw distance data
    image_matrix = np.zeros((camera_count, img_height_pixels, img_width_pixels, 3), np.uint8)
    depth_matrix = np.full((camera_count, img_height_pixels, img_width_pixels), -1, np.float64) # Fill with -1 as an invalid distance value

    depth_max = np.empty((camera_count))
    depth_min = np.empty((camera_count))
    depth_delta = np.empty((camera_count))
    for i in range(camera_count):
        depth_max[i] = max(screen_point_array[i][2])
        depth_min[i] = min(screen_point_array[i][2])
        depth_delta[i] = depth_max[i] - depth_min[i]

    # Paint the image
    for i in range(camera_count):
        for j in range(point_count):
            x_position = int(screen_point_array[i][0][j] * physical_to_pixels)
            y_position = int(screen_point_array[i][1][j] * physical_to_pixels)
            depth = screen_point_array[i][2][j]
            color = 255 * (1 - (depth - depth_min[i]) / (depth_delta[i])) # normalize, scale to 0-255, and invert
            if ((x_position < img_width_pixels and x_position >= 0) and (y_position < img_height_pixels and y_position >= 0)):
                if (image_matrix[i][y_position][x_position][0] < color):
                    image_matrix[i, y_position, x_position] = color
                    depth_matrix[i, y_position, x_position] = screen_point_array[i][2][j]

    # Save raw depth as csv
    for i in range(camera_count):
        np.savetxt(f"depth_matrix_{i}.csv", depth_matrix[i], delimiter=",", newline="\n")
        np.savetxt(f"extrinsic_matrix_{i}.csv", camera_extrinsic_matrix[i], delimiter=",", newline="\n")
        img = Image.fromarray(image_matrix[i])
        img.save(f"color_visualization_{i}.png")

if __name__ == "__main__":
    main()