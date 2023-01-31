import os
from PIL import Image


def split_image(img_path, new_dir, file_name):
    # Open the image
    img = Image.open(img_path)

    # Get the image size
    w, h = img.size

    # Calculate the new image size
    new_w = w // 4
    new_h = h // 4

    # Split the image into 4 parts
    imgs = [
        img.crop((i * new_w, j * new_h, (i + 1) * new_w, (j + 1) * new_h))
        for i in range(4)
        for j in range(4)
    ]

    # Save the parts
    for i, part in enumerate(imgs):
        part.save(os.path.join(new_dir, f"{file_name}_{i}.png"))


def process_directory(dir_path, new_dir):
    # Create the new directory if it doesn't exist
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)

    # Get all the files in the directory
    files = os.listdir(dir_path)

    # Split each image
    for file in files:
        # remove ".png" extension
        file_name = file[:-4]
        split_image(os.path.join(dir_path, file), new_dir, file_name)


# TODO: Use another variable for original_data path and new_data path for sake of reuseability

process_directory("./original_data/val/A", "./data/val/A")
process_directory("./original_data/val/B", "./data/val/B")
process_directory("./original_data/val/label", "./data/val/label")

process_directory("./original_data/test/A", "./data/test/A")
process_directory("./original_data/test/B", "./data/test/B")
process_directory("./original_data/test/label", "./data/test/label")

process_directory("./original_data/train/A", "./data/train/A")
process_directory("./original_data/train/B", "./data/train/B")
process_directory("./original_data/train/label", "./data/train/label")
