import glob
# import streamlit as st

from scipy.spatial.distance import cosine
import os

from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Model
import numpy as np

base_model = InceptionV3(weights='imagenet', include_top=False)
model = Model(inputs=base_model.input, outputs=base_model.output)


image_directory = 'outerwear/ponchos'
image_directory1 = 'outerwear/blazers'
image_directory2 = 'outerwear/coats'
image_directory3 = 'outerwear/jackets'
image_directory4 = 'outerwear/cloth'

direct = 'bottom/jeans'
direct1 = 'bottom/legging'
direct2 = 'bottom/pants'
direct3 = 'bottom/shorts'
direct4 = 'bottom/skirts'
direct5 = 'bottom/sweatpants'

# Define the maximum number of photos to take from each directory
max_photos_per_directory = 1000

# Initialize counters for each directory
counter = 0

# List to store image paths
image_paths_list = []

# Iterate over all directories
for directory in [image_directory, image_directory1, image_directory2, image_directory3, image_directory4,direct,direct1,direct2,direct3,direct4,direct5]:
    # Reset counter for each directory
    counter = 0
    # Iterate over all files in the directory
    for file in glob.glob(os.path.join(directory, '*.*')):
        # Check if the file is an image file
        if file.endswith(('.jpg', '.png', '.jpeg', 'webp')):
            # Append the file path to the list
            image_paths_list.append(file)
            # Increment the counter
            counter += 1
            # Check if the maximum number of photos per directory has been reached
            if counter >= max_photos_per_directory:
                break  # Exit the loop if the maximum number is reached

# Print the number of images loaded from each directory
# for i, directory in enumerate([image_directory, image_directory1, image_directory2, image_directory3, image_directory4]):
#     print(f"Number of images loaded from {directory}: {image_paths_list[i * max_photos_per_directory : (i + 1) * max_photos_per_directory]}")

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array_expanded = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array_expanded)

def extract_features(model, preprocessed_img):
    features = model.predict(preprocessed_img)
    flattened_features = features.flatten()
    normalized_features = flattened_features / np.linalg.norm(flattened_features)
    return normalized_features




def recommend_fashion_items_cnn(input_image_path, all_features, all_image_paths, input_category, top_n=6):
    # Pre-process the input image and extract features
    preprocessed_img = preprocess_image(input_image_path)
    input_features = extract_features(model, preprocessed_img)

     # Get the category of the input image
    # input_category = os.path.basename(os.path.dirname(input_image_path))
    # file_path = input_image_path
    # folders = file_path.split("/")  # Split the file path by "/"
    print(input_category)
    print(input_features)

    # Calculate similarities with all images
    similarities = []
    compatibilities=[]
    for feature, image_path in zip(all_features, all_image_paths):
        file_path = image_path
        folders = file_path.split("/")  # Split the file path by "/"
        category = folders[2]  # Get the third element from the list (index 3)
        print(category, input_category)

        if category == input_category:
            similarity = 1 - cosine(input_features, feature)
            similarities.append((image_path, similarity))
        if category != input_category:
          similarity = 1 - cosine(input_features, feature)
          compatibilities.append((image_path, similarity))

    # Sort the similarities based on similarity score
    similarities.sort(key=lambda x: x[1], reverse=True)
    compatibilities.sort(key=lambda x: x[1], reverse=True)


    # recommended_simi_image_paths = []  # Store recommended image paths
    # recommended_comp_image_paths = []  # Store recommended image paths

    recommended_simi_image_paths = set()  # Store recommended image paths to avoid duplicates for similar items
    recommended_comp_image_paths = set()  # Store recommended image paths to avoid duplicates for compatible items
    count_simi = 0
    count_comp = 0


    for image_path, similarity in similarities:
      if image_path != input_image_path:
        count_simi += 1

        if image_path not in recommended_simi_image_paths and image_path != input_image_path:
         recommended_simi_image_paths.add(image_path)  # Add path to the list
         with open("recommended_simi_image_paths.txt", 'w') as file:
          for path in recommended_simi_image_paths:
              file.write(path + '\n')
          if count_simi == top_n:
              break


    for image_path, similarity in compatibilities:
        count_comp += 1
        if image_path not in recommended_simi_image_paths and image_path != input_image_path:
         recommended_comp_image_paths.add(image_path)  # Add path to the list
         with open("recommended_comp_image_paths.txt", 'w') as file:
          for path in recommended_comp_image_paths:
              file.write(path + '\n')
          if count_comp == top_n:
              break