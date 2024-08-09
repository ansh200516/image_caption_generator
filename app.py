import streamlit as st
import json
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.models import Model

# Read the files word_to_idx.pkl and idx_to_word.pkl to get the mappings between word and index
word_to_index = {}
with open("D:\image_caption_generator\word_to_idx.pkl", 'rb') as file:
    word_to_index = pd.read_pickle(file)

index_to_word = {}
with open("D:\image_caption_generator\idx_to_word.pkl", 'rb') as file:
    index_to_word = pd.read_pickle(file)

# Load the model
model = load_model("D:\image_caption_generator\model_19.h5")

resnet50_model = ResNet50(weights='imagenet', input_shape=(224, 224, 3))
resnet50_model = Model(resnet50_model.input, resnet50_model.layers[-2].output)

# Generate Captions for an image
def predict_caption(photo):
    inp_text = "startseq"
    for i in range(38):
        sequence = [word_to_index.get(w, 0) for w in inp_text.split()]
        sequence = pad_sequences([sequence], maxlen=38, padding='post')
        ypred = model.predict([photo, sequence])
        ypred = ypred.argmax()
        word = index_to_word.get(ypred, '')
        inp_text += (' ' + word)
        if word == 'endseq':
            break
    final_caption = inp_text.split()[1:-1]
    final_caption = ' '.join(final_caption)
    return final_caption

def preprocess_image(img):
    img = image.load_img(img, target_size=(224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

def encode_image(img):
    img = preprocess_image(img)
    feature_vector = resnet50_model.predict(img)
    return feature_vector

st.title('Image Caption Generator')
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    photo = encode_image(uploaded_file).reshape((1, 2048))
    caption = predict_caption(photo)
    st.write("Caption: ", caption)
