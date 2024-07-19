import streamlit as st
import numpy as np
import json
from PIL import Image
from tensorflow import keras

# Load the trained model
model = keras.models.load_model("place_recommend_model.h5")

# Load class indices from JSON file
class_indices = json.load(open("class_indices.json"))

# Load place descriptions from JSON file
place_descriptions = json.load(open("description.json"))

# Function to preprocess image
def preprocess_image(image):
    image_rgb = image.convert("RGB")
    resized_image = image_rgb.resize((224, 224))
    normalized_image = np.array(resized_image, dtype=np.float32) / 255.0
    input_image = np.expand_dims(normalized_image, axis=0)
    return input_image

# Function to predict place
def predict_place(model, image, class_indices):
    preprocessed_img = preprocess_image(image)
    predictions = model.predict(preprocessed_img)[0]
    predicted_class_index = np.argmax(predictions)
    predicted_class_name = class_indices[str(predicted_class_index)]
    return predicted_class_name

# Function to get description
def get_description(predicted_place, place_descriptions):
    for place in place_descriptions:
        if place['place'] == predicted_place:
            return place['description_nepali'], place['description_english']
    return None, None

# Configure page settings
st.set_page_config(
    page_title="Dharohar(धरोहर)",
    page_icon=":place:",
    layout="centered"
)

st.title("Dharohar(धरोहर)")
st.write("An app dedicated to explore the rich cultural and historical heritage of the Far Western Province of Nepal.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Find out"):
        predicted_place = predict_place(model, image, class_indices)
        nepali_desc, english_desc = get_description(predicted_place, place_descriptions)
        
        # Store the results in session state
        st.session_state['predicted_place'] = predicted_place
        st.session_state['nepali_desc'] = nepali_desc
        st.session_state['english_desc'] = english_desc
        st.session_state['description_language'] = 'english'

# Check if we have results stored in session state
if 'predicted_place' in st.session_state:
    st.success(st.session_state['predicted_place'])

    if 'description_language' not in st.session_state:
        st.session_state['description_language'] = 'english'

    # Display the description based on the current language setting
    if st.session_state['description_language'] == "english":
        st.subheader("Description")
        st.subheader("English")
        st.write(st.session_state['english_desc'])
    else:
        st.subheader("Description")
        st.subheader("Nepali")
        st.write(st.session_state['nepali_desc'])
    
    # Toggle description language
    if st.session_state['description_language'] == "english":
        if st.button("Show in Nepali"):
            st.session_state['description_language'] = "nepali"
            st.rerun()
    else:
        if st.button("Show in English"):
            st.session_state['description_language'] = "english"
            st.rerun()
st.write("© 2024 Dharohar.All rights reserved")