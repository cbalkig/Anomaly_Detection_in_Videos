import os
import time
import concurrent.futures
from config import Config
import uuid
from PIL import Image


def save_image(image_class, img_file_name, img):
    img_folder = os.path.join(Config.EDIT_DATASET_PATH, image_class)
    if not os.path.exists(img_folder):
        os.makedirs(img_folder)
    img_file = os.path.join(img_folder, img_file_name)
    img.save(img_file)


def preprocess_image(sub_directory_path, file_name, class_name):
    img_path = os.path.join(sub_directory_path, file_name)
    if str(img_path)[-3:] == "tif":
        try:
            img = Image.open(img_path)
            img = img.resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE))
            new_file_name = str(uuid.uuid4()) + ".tif"
            save_image(class_name, new_file_name, img)
        except:
            print("Problem on file", img_path)


def preprocess_images():
    start_time = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        futures = []
        for f in sorted(os.listdir(Config.DATASET_PATH)):
            directory_path = os.path.join(Config.DATASET_PATH, f)
            if os.path.isdir(directory_path):
                class_name = f
                for v in sorted(os.listdir(directory_path)):
                    sub_directory_path = os.path.join(directory_path, v)
                    print("Processing", sub_directory_path)
                    for c in sorted(os.listdir(sub_directory_path)):
                        futures.append(
                            executor.submit(
                                preprocess_image, sub_directory_path=sub_directory_path, file_name=c,
                                class_name=class_name
                            )
                        )
    print("Execution time:", time.time() - start_time, "seconds.")


preprocess_images()
