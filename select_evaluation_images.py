import os
import utils
import pandas as pd
import numpy as np
import shutil

# load json with paths
config = utils.load_config_file("configs.json")
dataset_label = config['dataset_label']
dataset_path = config['paths']['images']
annotations_path = config['paths']['segments_annotations']
tests_path = f'evaluation/{dataset_label}/images'
os.makedirs(tests_path, exist_ok=True)

# training images to not use in test set
df = pd.read_csv(annotations_path)
training_img_paths = np.unique(df['img_path'].to_numpy())
training_img_files = [os.path.basename(path) for path in training_img_paths]

# load images from dataset folder
image_files = [f for f in os.listdir(dataset_path) if f.endswith(('png', 'jpg', 'jpeg'))]

# select test set
sample_size = 30
test_image_files = list(set(image_files) - set(training_img_files))
test_image_files = np.random.choice(image_files, sample_size, replace=False)

# save to tests folder
for file in test_image_files:
    src_path = os.path.join(dataset_path, file)
    dst_path = os.path.join(tests_path, file)
    shutil.copy(src_path, dst_path)