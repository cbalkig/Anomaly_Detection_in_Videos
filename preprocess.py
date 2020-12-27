import os
import time
import concurrent.futures
from config import Config
import uuid
from PIL import Image
import matplotlib.pyplot as plt


def tree(folder, max_items=10000000):
    print("Tree view of", folder)
    directory_count = 0
    for directory_path, directory_names, filenames in os.walk(folder):
        directory_level = directory_path.replace(folder, "")
        directory_level = directory_level.count(os.sep)
        indent = " " * 4
        print("{}{}/".format(indent * directory_level, os.path.basename(directory_path)))

        for i in range(len(filenames)):
            if i >= max_items:
                print("{}{}".format(indent * (directory_level + 1), "..."))
                break
            print("{}{}".format(indent * (directory_level + 1), filenames[i]))

        directory_count = directory_count + 1
        if directory_count >= max_items:
            print("{}{}/".format(indent * directory_level, "..."))
            break


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
                    #print("Processing", sub_directory_path)
                    for c in sorted(os.listdir(sub_directory_path)):
                        futures.append(
                            executor.submit(
                                preprocess_image, sub_directory_path=sub_directory_path, file_name=c,
                                class_name=class_name
                            )
                        )
    print("Execution time:", time.time() - start_time, "seconds.")


def get_stats():
    total_samples_count = 0
    for f in sorted(os.listdir(Config.EDIT_DATASET_PATH)):
        directory_path = os.path.join(Config.EDIT_DATASET_PATH, f)
        if os.path.isdir(directory_path):
            class_name = f
            sample_count = len(os.listdir(directory_path))
            print("Class Name:", class_name, ", Samples Count:", sample_count)
            total_samples_count = total_samples_count + sample_count
            plt.figure(figsize=(10, 10))
            plt.title("Some examples of class " + class_name)
            count = 0
            images = []
            for v in sorted(os.listdir(directory_path)):
                img_path = os.path.join(directory_path, v)
                img = Image.open(img_path)
                images.append(img)
                count = count + 1
                if count >= 25:
                    break
            for n in range(25):
                ax = plt.subplot(5, 5, n + 1)
                plt.imshow(images[n], cmap='gray')
                plt.axis('off')
            plt.show()
    print("Total Samples Count:", total_samples_count)


tree(Config.DATASET_PATH, max_items=3)
preprocess_images()
tree(Config.EDIT_DATASET_PATH, max_items=3)
get_stats()
