import numpy as np
from PIL import Image

def main():
    option_list = ["compressed", "uncompressed"]
    img = np.zeros((2, 576, 640), np.float64)
    for index, option in enumerate(option_list):
        img_low = Image.open(f"{option}_depth_low.png")
        img_high = Image.open(f"{option}_depth_high.png")
        img_low = np.asarray(img_low)
        img_high = np.asarray(img_high)
        img_low = img_low[:, :, 2]
        img_high = img_high[:, :, 2]
        img[index] = join_low_high_matrices(img_low, img_high)
        print("Worked")

    mse = mse_between_two_images(img[0], img[1])
    print(f"The MSE between the compressed and uncompressed depth images is {mse}")

def join_low_high_matrices(low_image, high_image):
    float16_image = np.zeros((low_image.shape[0], low_image.shape[1]), np.float16)

    for y_index, row in enumerate(low_image):
        for x_index, low_pixel in enumerate(row):
            high_pixel = high_image[y_index][x_index]
            if (low_pixel != 0 or high_pixel != 0):
                int16_pixel = (high_pixel << 8) | low_pixel
                float16_pixel = np.uint16(int16_pixel).view(np.float16)
                float16_image[y_index][x_index] = float16_pixel
                # print(f"lp, hp, fp: {low_pixel, high_pixel, float16_pixel}")
        
    return float16_image

def mse_between_two_images(prediction_image, target_image):
    height = prediction_image.shape[0]
    width = prediction_image.shape[1]
    total_sum = 0
    dataset_size = height * width

    for y in range(height):
        for x in range(width):
            prediction_delta = target_image[y][x] - prediction_image[y][x]
            print(f"ti: {target_image[y][x]}, pi: {prediction_image[y][x]}, pd: {prediction_delta}, pd2: {prediction_delta**2}")
            total_sum += (prediction_delta)**2

    mse = total_sum/float(dataset_size)
    return mse

if __name__ == "__main__":
    main()