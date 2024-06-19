import sys
import numpy as np
import open3d as o3d

def save_to_point_cloud(point_array, filename):
<<<<<<< HEAD
    new_point_array = np.empty((point_array.shape[0], 3), np.float64)
=======
    new_point_array = np.zeros((point_array.size, 3), np.float64)
>>>>>>> d686c239165eec0bcfc3885e563803a7700bbbf0
    
    for index, point in enumerate(point_array):
        new_point_array[index][0] = point[0]
        new_point_array[index][1] = point[1]
        new_point_array[index][2] = point[2]

    new_point_cloud = o3d.geometry.PointCloud()
    new_point_cloud.points = o3d.utility.Vector3dVector(new_point_array)
    o3d.io.write_point_cloud(filename, new_point_cloud)
    

def read_matrix(filename):
    matrix = np.loadtxt(open(filename, "rb"), delimiter = ",")
    return matrix

def transformed_point_cloud(point_array, transformation_matrix):
<<<<<<< HEAD
    new_point_array = np.empty((point_array.shape[0], 4), np.float64)
=======
    new_point_array = np.ones((point_array.size, 4), np.float64)
>>>>>>> d686c239165eec0bcfc3885e563803a7700bbbf0

    for index, point in enumerate(point_array):
        new_point = transformation_matrix @ point
        new_point_array[index] = new_point

    return new_point_array

def inverse_pinhole(screen_point, img_width_pixels, img_height_pixels, row, column):
    focal_length_cm = 0.1
    img_width_cm = 0.24
    img_height_cm = 0.24
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

def screen_to_world(depth_matrix):
    img_height_pixels, img_width_pixels = depth_matrix.shape

    # Loop through the depth_matrix and figure out how many valid distance pixels there are (-1 is invalid)
    point_counter = 0
    for row in depth_matrix:
        for point in row:
            if point > 0:
                point_counter += 1

    print("There are ", point_counter, " valid points") # Makes sure we are only counting the valid points

    point_array = np.ones((point_counter, 4), np.float64) # Create an array only fit for the valid points. first column is for x, second is for y, third for z, and fourth for homogenous linear algebra

    # Loop through depth_matrix again, but only process valid points
    index = 0
    for row_index, row in enumerate(depth_matrix):
        for col_index, point in enumerate(row):
            if point > 0:
                point_array[index] = inverse_pinhole(point, img_width_pixels, img_height_pixels, row_index, col_index)
                index += 1
            
    return point_array


def main():
    if len(sys.argv) < 2 or not sys.argv[1].isdigit():
        print("Usage: python image_to_pointcloud.py <camera_count>")
        return

    camera_count = int(sys.argv[1])
    
    for i in range(camera_count):
        depth_matrix = read_matrix(f"depth_matrix_{i}.csv") # Loads csv as 2D numpy array
        original_extrinsic_matrix = read_matrix(f"extrinsic_matrix_{i}.csv")

        point_array_original = screen_to_world(depth_matrix)
        inverse_original_extrinsic_matrix = np.linalg.inv(original_extrinsic_matrix)

        point_array_world = transformed_point_cloud(point_array_original, inverse_original_extrinsic_matrix)
        
        save_to_point_cloud(point_array_original, f"original_point_cloud_{i}.ply")
        save_to_point_cloud(point_array_world, f"visible_point_cloud_{i}.ply")


if __name__ == "__main__":
    main() 