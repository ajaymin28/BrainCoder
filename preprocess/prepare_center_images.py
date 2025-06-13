import os
from glob import glob

# Change these paths as needed
object_images_root = '/home/ja882177/EEG/gits/NICE-EEG/Data/Things-EEG2/Image_set/center_images'
test_images_root = '/home/ja882177/EEG/gits/NICE-EEG/Data/Things-EEG2/Image_set/test_images'

# 1. Collect all filenames in test_images (regardless of subfolder)
test_filenames = set()
for dirpath, dirnames, filenames in os.walk(test_images_root):
    for fname in filenames:
        test_filenames.add(fname)

print(f"Found {len(test_filenames)} test images to discard.")

# 2. Loop through all images in object_images and remove if in test_filenames
removed = 0
for obj_category in os.listdir(object_images_root):
    category_path = os.path.join(object_images_root, obj_category)
    if not os.path.isdir(category_path):
        continue
    for fname in os.listdir(category_path):
        if fname in test_filenames:
            file_to_remove = os.path.join(category_path, fname)
            if os.path.exists(file_to_remove):
                os.remove(file_to_remove)
                removed += 1
                print(f"Removed: {file_to_remove}")

print(f"\nTotal images removed: {removed}")
