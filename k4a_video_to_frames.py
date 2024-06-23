import numpy as np
from PIL import Image

def main():
    # Use 'ffmpeg -i ..\room_test.mkv -map 0:1 -vf "select='eq(n\,0)'" -vsync vfr -frames:v 1 -f rawvideo output.raw' to grab a frame from mkv
    width, height = 640, 576

    high_byte, low_byte = read_raw_data("output.raw", width, height)
    img = create_image_from_data(high_byte, low_byte, width, height)

    img.save("output.png")
    print("did it")

def read_raw_data(file_path, width, height):
    with open(file_path, "rb") as file:
        raw_data = file.read()

    float_data = np.frombuffer(raw_data, dtype=">f2")

    float_data = float_data.reshape((height, width))

    high_byte = (float_data.view('>u2') >> 8).astype(np.uint8)
    low_byte = (float_data.view('>u2') & 0xFF).astype(np.uint8)

    return high_byte, low_byte

def create_image_from_data(high_byte, low_byte, width, height):
    img = np.zeros((height, width, 3), dtype=np.uint8)

    img[:, :, 1] = high_byte
    img[:, :, 2] = low_byte

    img = Image.fromarray(img)

    return img

if __name__ == "__main__":
    main()