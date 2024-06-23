import os, sys
import numpy as np
from PIL import Image, ImageDraw
import open3d as o3d

depth_img_width_cm = 0.24
depth_img_height_cm = 0.24
color_img_width_cm = 0.24
color_img_height_cm = 0.24

focal_length_depth_cm = 0.1

depth_img_width_pixels = 1024
depth_img_height_pixels = 1024
color_img_width_pixels = 1024
color_img_height_pixels = 1024

physical_to_pixels_depth = depth_img_width_pixels / depth_img_width_cm

def main():
    if len(sys.argv) < 3 or not sys.argv[2].isdigit():
        print("Usage: python capture.py <point_cloud_filename> <camera_count>")
        return

    script_location = os.path.dirname(os.path.realpath(__file__))
    camera_count = int(sys.argv[2])

    # Loads point cloud
    point_load = read_point_cloud(f"{script_location}/point_clouds/{sys.argv[1]}")
    if point_load == None:
        return
    point_count = len(point_load.points)
    point_coordinate_array = np.asarray(point_load.points)
    point_color_array = np.asarray(point_load.colors)
    
    #  Define extrinsic matrix for each camera
    depth_camera_extrinsic_matrix = np.empty((camera_count, 4, 4), np.float16)
    color_camera_extrinsic_matrix = np.empty((camera_count, 4, 4), np.float16)
    for i in range(camera_count):
        current_z_rotation = 360 / camera_count * i
        depth_camera_extrinsic_matrix[i] = extrinsic_matrix(0, -600, 1200, 0, current_z_rotation, 0) # Get a 360 view from <camera_count> cameras
        color_camera_extrinsic_matrix[i] = extrinsic_matrix(0, -600, 1200, 0, current_z_rotation, 0) # Get a 360 view from <camera_count> cameras
        print(f"[DEBUG] Camera[{i}] extrinsic matrix:\n", depth_camera_extrinsic_matrix[i])

    depth_matrix_low = np.zeros((camera_count, depth_img_height_pixels, depth_img_width_pixels, 3), np.uint8) # to png
    depth_matrix_high = np.zeros((camera_count, depth_img_height_pixels, depth_img_width_pixels, 3), np.uint8) # to png
    depth_matrix_float16 = np.zeros((camera_count, depth_img_height_pixels, depth_img_width_pixels), np.float16) # to csv
    color_matrix = np.zeros((camera_count, color_img_height_pixels, color_img_width_pixels, 3), np.uint8) # to png

    screen_point_depth_array = rasterize_points_to_matrix(camera_count, depth_camera_extrinsic_matrix, point_coordinate_array)
    screen_point_color_array = rasterize_points_to_matrix(camera_count, color_camera_extrinsic_matrix, point_coordinate_array, point_color_array)

    # Paint the images
    for i in range(camera_count):
        for j in range(point_count):
            x_position = int(screen_point_depth_array[i][0][j] * physical_to_pixels_depth)
            y_position = int(screen_point_depth_array[i][1][j] * physical_to_pixels_depth)
            depth_float16 = screen_point_depth_array[i][2][j]
            depth_low, depth_high = float16_to_two_int8(depth_float16)
            if ((x_position < depth_img_width_pixels and x_position >= 0) and (y_position < depth_img_height_pixels and y_position >= 0)):
                # Check if current point is closer than the one currently at the pixel or if the pixel is blank
                if (depth_matrix_float16[i][y_position][x_position] > depth_float16 or abs(depth_matrix_float16[i][y_position][x_position]) < 0.1): 
                    depth_matrix_low[i, y_position, x_position, 2] = depth_low
                    depth_matrix_high[i, y_position, x_position, 2] = depth_high
                    depth_matrix_float16[i, y_position, x_position] = depth_float16

                    color_matrix[i, y_position, x_position] = np.asarray([screen_point_color_array[i][2][j], screen_point_color_array[i][3][j], screen_point_color_array[i][4][j]])

    # Save raw depth as csv
    for i in range(camera_count):
        np.savetxt(f"depth_matrix_float16_{i}.csv", depth_matrix_float16[i], delimiter=",", newline="\n")
        np.savetxt(f"depth_camera_extrinsic_matrix_{i}.csv", depth_camera_extrinsic_matrix[i], delimiter=",", newline="\n")
        np.savetxt(f"color_camera_extrinsic_matrix_{i}.csv", color_camera_extrinsic_matrix[i], delimiter=",", newline="\n")
        img_low = Image.fromarray(depth_matrix_low[i])
        img_high = Image.fromarray(depth_matrix_high[i])
        img_color = Image.fromarray(color_matrix[i])
        img_low.save(f"depth_visualization_low_{i}.png")
        img_high.save(f"depth_visualization_high_{i}.png")
        img_color.save(f"color_visualization_{i}.png")

def rasterize_points_to_matrix(camera_count, depth_camera_extrinsic_matrix, point_coordinate_array, point_color_array=np.empty((0, 0))):
    point_count = point_coordinate_array.shape[0]
    # Create progress markers
    marker_counter = 4
    progress_markers = create_progress_markers(marker_counter, point_count)
    marker_index = 0
    print(progress_markers)

    if (point_color_array.shape[0] != 0): # True if color, false if depth capture
        color = True
    else:
        color = False

    print(color)
    print(point_color_array)

    if (color == True):
        screen_point_array = np.empty((camera_count, 5, point_count), np.uint8) # For each camera, store x, y, and rgb info
    else:
        screen_point_array = np.empty((camera_count, 3, point_count), np.float16) # For each camera, store x, y, and distance of each point in screen cm coordinates

    # Loop through point cloud and add rasterized points to an array
    for camera_index in range(camera_count):
        for point_index, world_point in enumerate(point_coordinate_array):
            world_point_homogeneous = np.array([[world_point[0]],
                                                [world_point[1]],
                                                [world_point[2]],
                                                [1]])
            camera_point = depth_camera_extrinsic_matrix[camera_index] @ world_point_homogeneous

            if (color == True):
                raster_point = pinhole_projection(camera_point, point_color_array[point_index])
                screen_point_array[camera_index][0][point_index] = raster_point[0] # Screen X (cm)
                screen_point_array[camera_index][1][point_index] = raster_point[1] # Screen Y (cm)
                screen_point_array[camera_index][2][point_index] = raster_point[2] # Point R
                screen_point_array[camera_index][3][point_index] = raster_point[3] # Point G
                screen_point_array[camera_index][4][point_index] = raster_point[4] # Point B
            else:
                raster_point = pinhole_projection(camera_point)
                screen_point_array[camera_index][0][point_index] = raster_point[0] # Screen X (cm)
                screen_point_array[camera_index][1][point_index] = raster_point[1] # Screen Y (cm)
                screen_point_array[camera_index][2][point_index] = raster_point[2] # Point distance (cm)
            
            if (point_index == progress_markers[marker_index]):
                counter_percentage = (marker_index + 1) / (marker_counter + 1) * 100
                print(f"point {point_index} successfully processed for camera {camera_index}. {counter_percentage}% done")
                marker_index += 1
                marker_index %= marker_counter
        print(f"Every point has been successfully processed for camera {camera_index}. 100% done")

    return screen_point_array

def create_progress_markers(marker_count, total_to_process):
    progress_markers = np.empty((marker_count))
    for i in range(marker_count):
        progress_markers[i] = int(total_to_process / (marker_count + 1) * (i + 1))
    
    return progress_markers

def float16_to_two_int8(float16):
    int16 = np.float16(float16).view(np.uint16) # Convert to int16 while keeping the bits intact

    low_int8 = int16 & 0x00FF
    high_int8 = (int16 >> 8) & 0x00FF

    return low_int8, high_int8

def read_point_cloud(file_path):
    point_cloud = o3d.io.read_point_cloud(file_path)
    
    if len(point_cloud.points) == 0:
        print(f"Failed to read point cloud from '{file_path}'")
        return None
    else:
        return point_cloud

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

    depth_camera_extrinsic_matrix = np.array([[n11, n12, n13, camera_position_x],
                                         [n21, n22, n23, camera_position_y],
                                         [n31, n32, n33, camera_position_z],
                                         [  0,   0,   0,                 1]])

    return depth_camera_extrinsic_matrix

def pinhole_projection(point_3d, rgb_info=np.empty((0,0))):
    """
    Project a 3D point onto a 2D image plane using the pinhole camera model.
    
    Arg:
    - point_3d: A numpy array representing the 3D point in Cartesian coordinates (x, y, z).
        
    Returns:
    - A numpy array representing the 2D projection of the 3D point on the image plane.
    """
    
    # Project the 3D point onto the image plane
    projection_x = -(focal_length_depth_cm * point_3d[0][0]) / (point_3d[2][0]) # Invert signal because of projection properties
    projection_y = -(focal_length_depth_cm * point_3d[1][0]) / (point_3d[2][0])
    # Convert to pixel coordinates
    pixel_x = (projection_x + depth_img_width_cm / 2)
    pixel_y = (projection_y + depth_img_height_cm / 2)

    if(rgb_info.shape[0] == 0):
        projection_h = np.sqrt(point_3d[0][0] ** 2 + point_3d[1][0] ** 2)
        pixel_depth = np.sqrt(projection_h ** 2 + point_3d[2][0] ** 2)
        return np.array([pixel_x, pixel_y, pixel_depth])

    else:
        red, green, blue = rgb_info[0], rgb_info[1], rgb_info[2]
        # Necessary because color values are stored as floats ranging from 0 to 1
        red *= 255
        green *= 255
        blue *= 255
        return np.array([pixel_x, pixel_y, red, green, blue], np.uint8)
            


if __name__ == "__main__":
    main()