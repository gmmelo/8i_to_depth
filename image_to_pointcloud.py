import sys
import numpy as np
import open3d as o3d
from PIL import Image

def main():
    if len(sys.argv) < 2 or not sys.argv[1].isdigit():
        print("Usage: python image_to_pointcloud.py <camera_count>")
        return

    camera_count = int(sys.argv[1])
    
    for i in range(camera_count):
        # Loads two 8-bit int images as a 16-bit numpy matrix containing depth info
        depth_matrix = read_low_high_image(f"output.png")
        color_matrix = read_image_as_matrix(f"test_calibration_color_{i}.png")

        depth_camera_extrinsic_matrix = read_matrix(f"depth_camera_extrinsic_matrix_{i}.csv")
        color_camera_extrinsic_matrix = read_matrix(f"color_camera_extrinsic_matrix_{i}.csv")

        point_coordinate_array_original, point_color_array = screen_to_world(depth_matrix, color_matrix)
        inverse_depth_camera_extrinsic_matrix = np.linalg.inv(depth_camera_extrinsic_matrix)

        point_coordinate_array_world = transformed_point_cloud(point_coordinate_array_original, inverse_depth_camera_extrinsic_matrix)
        
        save_to_point_cloud(point_coordinate_array_original, point_color_array, f"original_point_cloud_{i}.ply")
        save_to_point_cloud(point_coordinate_array_world, point_color_array, f"transformed_point_cloud_{i}.ply")

def save_to_point_cloud(point_coordinate_array, point_color_array, filename):
    new_point_coordinate_array = np.empty((point_coordinate_array.shape[0], 3), np.float16)
    
    for index, point in enumerate(point_coordinate_array):
        new_point_coordinate_array[index][1] = point[1]
        new_point_coordinate_array[index][2] = point[2]
        new_point_coordinate_array[index][0] = point[0]

    new_point_cloud = o3d.geometry.PointCloud()
    new_point_cloud.points = o3d.utility.Vector3dVector(new_point_coordinate_array)
    new_point_cloud.colors = o3d.utility.Vector3dVector(point_color_array)
    o3d.io.write_point_cloud(filename, new_point_cloud)
    

def read_matrix(filename):
    matrix = np.loadtxt(open(filename, "rb"), delimiter = ",")
    return matrix

def read_low_high_image(filename):
    image = read_image_as_matrix(filename)
    low_image = image[:, :, 2]
    high_image = image[:, :, 1]
    float64_image = np.zeros((low_image.shape[0], low_image.shape[1]), np.float64)

    for y_index, row in enumerate(low_image):
        for x_index, low_pixel in enumerate(row):
            high_pixel = high_image[y_index][x_index]
            if (low_pixel != 0 or high_pixel != 0):
                int16_pixel = (high_pixel << 8) | low_pixel
                float64_pixel = np.uint16(int16_pixel).view(np.float16) * (2**24)
                float64_image[y_index][x_index] = float64_pixel
                # print(f"lp, hp, fp: {low_pixel, high_pixel, float16_pixel}")
            
            
    return float64_image

def read_image_as_matrix(filename):
    # This reads the image's blue channel and saves it to a brightness single channel matrix
    rgb_image = Image.open(filename)
    rgb_matrix = np.asarray(rgb_image)

    print(f"rgb image shape: {rgb_matrix.shape}")
    if (rgb_matrix.shape[2] == 4):
        rgb_matrix = np.delete(rgb_matrix, (3), axis=2)
    print(f"rgb image shape: {rgb_matrix.shape}")

    
    output_matrix = np.empty((rgb_image.height, rgb_image.width, 3), rgb_matrix.dtype)
    print(f"image_dtype: {rgb_matrix.dtype}")
    for row_index, row in enumerate(output_matrix):
        for column_index, pixel in enumerate(row):
            pixel = rgb_matrix[row_index][column_index]
            output_matrix[row_index][column_index] = pixel
    
    return output_matrix

    

def transformed_point_cloud(point_array, transformation_matrix):
    new_point_array = np.empty((point_array.shape[0], 4), np.float16)

    for index, point in enumerate(point_array):
        new_point = transformation_matrix @ point
        new_point_array[index] = new_point

    return new_point_array

def inverse_pinhole(screen_point, img_width_pixels, img_height_pixels, row, column):
    focal_length_cm = 0.2
    img_width_cm = 0.24
    img_height_cm = 0.216
    px_to_cm_x = img_width_cm / float(img_width_pixels)
    px_to_cm_y = img_height_cm / float(img_height_pixels)

    screen_center_y_cm = img_height_cm / 2
    screen_center_x_cm = img_width_cm / 2
    
    screen_x_cm = column * px_to_cm_x - screen_center_x_cm
    screen_y_cm = row * px_to_cm_y - screen_center_y_cm

    hipotenuse_screen = np.sqrt(screen_y_cm**2 + screen_x_cm**2)
    distance_screen = np.sqrt(focal_length_cm**2 + hipotenuse_screen**2)

    screen_world_ratio = screen_point / distance_screen

    world_x = -screen_x_cm * screen_world_ratio
    world_y = -screen_y_cm * screen_world_ratio
    world_z = focal_length_cm * screen_world_ratio

    world_point = np.asarray([world_x, world_y, world_z, 1])

    return world_point

def screen_to_world(depth_matrix, color_matrix):
    img_height_pixels, img_width_pixels = depth_matrix.shape

    # Loop through the depth_matrix and figure out how many valid distance pixels there are (-1 is invalid)
    point_counter = 0
    for row in depth_matrix:
        for point in row:
            if point > 0:
                point_counter += 1

    print("There are ", point_counter, " valid points") # Makes sure we are only counting the valid points

    point_coordinate_array = np.empty((point_counter, 4), np.float16) # Create an array only fit for the valid points. first column is for x, second is for y, third for z, and fourth for homogenous linear algebra
    point_color_array = np.empty((point_counter, 3), np.float16) # RGB info

    # Loop through depth_matrix again, but only process valid points
    index = 0
    for row_index, row in enumerate(depth_matrix):
        for col_index, depth_pixel in enumerate(row):
            if depth_pixel > 0:
                red = color_matrix[row_index, col_index, 0] / float(255)
                green = color_matrix[row_index, col_index, 1] / float(255)
                blue = color_matrix[row_index, col_index, 2] / float(255)
                color_pixel = np.array([red, green, blue])

                point_coordinate_array[index] = inverse_pinhole(depth_pixel, img_width_pixels, img_height_pixels, row_index, col_index)
                point_color_array[index] = color_pixel
                index += 1
            
    return point_coordinate_array, point_color_array


if __name__ == "__main__":
    main() 