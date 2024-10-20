import os
import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image

# Load the pre-trained models
category_model = tf.keras.models.load_model('models/categoryall.h5')
sleeves_model = tf.keras.models.load_model('models/sleevestshirt.h5')
color_model = tf.keras.models.load_model('models/colormodel.h5')
saree_model = tf.keras.models.load_model('models/sareecategory.h5')
gender_model = tf.keras.models.load_model('models/menwomenkid.h5')

# Define the class labels for each model
category_labels = ['jeans', 'saree', 'tshirt']
sleeves_labels = ['half sleeves', 'full sleeves', 'sleeveless']
color_labels = ['Black', 'Blue', 'Brown', 'Green', 'Violet', 'White', 'orange', 'red', 'yellow']
saree_labels = ['Banarasi', 'Bandhani', 'Ikat', 'Pichwai']
gender_labels = ['kid', 'men', 'women']


# Function to preprocess the image
def preprocess_image(image_path, target_size=(150, 150)):
    img = load_img(image_path, target_size=target_size)
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0
    return img


# Predict category
def predict_category(image):
    preds = category_model.predict(image)
    return category_labels[np.argmax(preds)]


# Predict color
def predict_color(image):
    preds = color_model.predict(image)
    return color_labels[np.argmax(preds)]


# Predict sleeve type for "tshirt"
def predict_sleeves(image):
    preds = sleeves_model.predict(image)
    return sleeves_labels[np.argmax(preds)]


# Predict gender for "tshirt"
def predict_gender(image):
    preds = gender_model.predict(image)
    return gender_labels[np.argmax(preds)]


# Predict saree type
def predict_saree(image):
    preds = saree_model.predict(image)
    return saree_labels[np.argmax(preds)]


# Display images from a folder with a heading
def display_images(folder_path, heading):
    # Check if the folder exists
    if not os.path.exists(folder_path):
        st.error(f"Folder {folder_path} does not exist.")
        return

    # Display all images in the folder
    image_files = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]

    if len(image_files) == 0:
        st.warning(f"No images found in the folder {heading}.")
        return

    # Display heading
    st.subheader(heading)

    # Create 5 columns for displaying images
    cols = st.columns(5)

    for idx, image_file in enumerate(image_files[:5]):  # Display the first 5 images
        img_path = os.path.join(folder_path, image_file)
        img = Image.open(img_path)
        cols[idx % 5].image(img, use_column_width=True, caption=image_file)


# Function to display images from the specified folder and its subfolders
def display_images_from_subfolders(base_folder):
    # Check if the base folder exists
    if not os.path.exists(base_folder):
        st.error(f"Base folder {base_folder} does not exist.")
        return

    # List all subfolders in the base folder
    subfolders = [f for f in os.listdir(base_folder) if os.path.isdir(os.path.join(base_folder, f))]

    # Limit to the first 5 subfolders
    subfolders = subfolders[:5]

    if not subfolders:
        st.warning("No subfolders found.")
        return

    # Display images from each subfolder
    for subfolder in subfolders:
        subfolder_path = os.path.join(base_folder, subfolder)
        display_images(subfolder_path, heading=subfolder)


# Main function to get combined prediction
def get_combined_prediction(image_path):
    image = preprocess_image(image_path)
    category = predict_category(image)

    if category == 'tshirt':
        sleeves = predict_sleeves(image)
        color = predict_color(image)
        gender = predict_gender(image)
        result = f"{category}-{sleeves}-{color}-{gender}"

    elif category == 'saree':
        saree_type = predict_saree(image)
        color = predict_color(image)
        result = f"{category}-{saree_type}-{color}"

    elif category == 'jeans':
        color = predict_color(image)
        result = f"{category}-{color}"

    else:
        result = f"Unknown category: {category}"

    return result


# Streamlit UI
st.title("Fashion Recommendation")

# Upload an image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Create the 'temp' directory if it doesn't exist
    temp_dir = "temp"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    # Save the uploaded file in the 'temp' directory
    temp_file = os.path.join(temp_dir, uploaded_file.name)
    with open(temp_file, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Display the uploaded image
    st.image(temp_file, caption='Uploaded Image.', use_column_width=True)

    # Get prediction
    result = get_combined_prediction(temp_file)
    st.write(f"Prediction: {result}")

    # Extract prediction components for folder path
    prediction_parts = result.split('-')

    # Define base folder path based on prediction
    if prediction_parts[0] == 'tshirt':
        base_folder = f"DataCategory/tshirt/{prediction_parts[1]}/{prediction_parts[2]}/{prediction_parts[3]}"
    elif prediction_parts[0] == 'saree':
        base_folder = f"DataCategory/saree/{prediction_parts[1]}/{prediction_parts[2]}"
    elif prediction_parts[0] == 'jeans':
        base_folder = f"DataCategory/jeans/{prediction_parts[1]}"
    else:
        st.write("Category not recognized.")
        base_folder = None

    if base_folder:
        # Display images from subfolders
        display_images_from_subfolders(base_folder)
