import numpy as np
from PIL import Image

def main():
    frame_list = ["depth_high", "depth_low"]
    for frame_name in frame_list:
        compressed_image = Image.open(f"compressed_{frame_name}.png")
        uncompressed_image = Image.open(f"uncompressed_{frame_name}.png")
        mse = mse_between_two_images(compressed_image, uncompressed_image)

        print(f"The MSE for the {frame_name} compression is {mse} for the blue channel")
        

def mse_between_two_images(prediction_image, target_image):
    height = prediction_image.height
    width = prediction_image.width
    total_sum = 0
    dataset_size = height * width

    for y in range(height):
        for x in range(width):
            prediction_delta = target_image.getpixel((x, y))[2] - prediction_image.getpixel((x, y))[2]
            total_sum += (prediction_delta)**2

    return total_sum/float(dataset_size)

if __name__ == "__main__":
    main()