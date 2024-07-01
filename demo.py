import streamlit as st
import torch
import clip
import torch
import torch.nn.functional as F
import pickle
import numpy as np
import PIL.Image

from src.data_utils import targetpad_transform
from src.combiner import Combiner


clip_model, clip_preprocess = clip.load('RN50x4')
clip_model.eval()

input_dim = clip_model.visual.input_resolution
preprocess = targetpad_transform(1.25, input_dim)

projection_dim = 2560
hidden_dim = 5120
feature_dim = clip_model.visual.output_dim
combiner = Combiner(feature_dim, projection_dim, hidden_dim)
combiner_state_dict = torch.load('fiq_comb_RN50x4_fullft.pt', map_location=torch.device('cpu'))
combiner.load_state_dict(combiner_state_dict['Combiner'])
combiner.eval()
combining_function = combiner.combine_features

FEATURE_PATH = 'demo_feature'
IMAGE_PATH = 'Dataset\fashionIQ_dataset\images'

def load_names(dress_type):
    with open(f'{FEATURE_PATH}\\{dress_type}_val_index_names.pkl', 'rb') as f:
        val_names = pickle.load(f)
    with open(f'{FEATURE_PATH}\\{dress_type}_test_index_names.pkl', 'rb') as f:
        test_names = pickle.load(f)

    return val_names + test_names

def load_features(dress_type):
    val_features = torch.load(f'{FEATURE_PATH}\\{dress_type}_val_index_features.pt')
    test_features = torch.load(f'{FEATURE_PATH}\\{dress_type}_test_index_features.pt')

    return torch.vstack((val_features, test_features))

def retrieve_image(dress, image, text, n_retrieved):
    names = load_names(dress)
    features = load_features(dress)
    features = F.normalize(features, dim=-1).float()
    
    img = preprocess(image)
    img = img.unsqueeze(0)

    text_tokens = clip.tokenize(text, truncate=True)
    
    with torch.no_grad():
        image_feature = clip_model.encode_image(img)
        text_feature = clip_model.encode_text(text_tokens)
        query_feature = combining_function(image_feature, text_feature).squeeze(0)

    query_feature = F.normalize(query_feature, dim=-1).float()

    index_features = F.normalize(features)
    cos_similarity = index_features @ query_feature.T
    sorted_indices = torch.topk(cos_similarity, n_retrieved, largest=True).indices.cpu()
    sorted_index_names = np.array(names)[sorted_indices].flatten()

    return sorted_index_names


dress_options = ['dress', 'shirt', 'toptee']
dress_type = st.selectbox('Choose dress type: ', dress_options)

image_query = st.file_uploader("Image query", type=['jpg', 'jpeg', 'png'])
if image_query:
    st.image(image_query, width=300)
    image_query = PIL.Image.open(image_query)
    image_query = image_query.resize((512, 512), PIL.Image.BICUBIC)

text_request = st.text_input("Text request: ")

n_retrieved = st.number_input("Number of retrieved images: ", min_value=0, max_value=100)

retrieve_button = st.button("Retrieve!")

if retrieve_button:
    retrieved_images = retrieve_image(dress_type, image_query, text_request, n_retrieved)

    for i in range(len(retrieved_images)):
        image = retrieved_images[i]
        image_path = f'{IMAGE_PATH}\\{image}.jpg'
        st.image(image_path, caption=f'{i}\t{image}',  width=300)
