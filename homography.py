import numpy as np

def inverse_pinhole(screen_point, img_width_pixels, img_height_pixels, row, column):
    focal_length_cm = 0.1
    img_width_cm = 0.24
    img_height_cm = 0.24

    world_point = screen_point # TODO: Calculate pinhole projection

    return world_point

def screen_to_world(depth_matrix):
    img_height_pixels, img_width_pixels = depth_matrix.shape

    # Loop through the depth_matrix and figure out how many valid distance pixels there are (-1 is invalid)
    point_counter = 0
    for column in depth_matrix:
        for point in column:
            if point != -1: # Might need to add delta for equality assertion with float
                point_counter += 1

    print("There are ", point_counter, " valid points") # Makes sure we are only counting the valid points

    point_array = np.empty((point_counter, 3), np.float64) # Create an array only fit for the valid points. first row is for x, second is for y, and third for z

    # Loop through depth_matrix again, but only process valid points
    index = 0
    for col_index, column in enumerate(depth_matrix):
        for row_index, point in enumerate(column):
            if point != -1:
                point_array[index] = inverse_pinhole(point, img_width_pixels, img_height_pixels, row_index, col_index)
                index += 1
            
    return point_array


def main():
    depth_matrix = np.loadtxt(open("depth_matrix.csv", "rb"), delimiter=",") # Loads csv as 2D numpy array
    point_array = screen_to_world(depth_matrix)
    print("Point array's shape: ", point_array.shape)

if __name__ == "__main__":
    main() 