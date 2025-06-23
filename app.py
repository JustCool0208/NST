import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from PIL import Image
import io

# Streamlit page setup
st.set_page_config(page_title="Neural Style Transfer", layout="wide")
st.title("🎨 Neural Style Transfer")
st.markdown("Transform your image using the style of another image using deep learning!")

# Load model from TensorFlow Hub (cached)
@st.cache_resource
def load_model():
    return hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

model = load_model()

# Load and process image
def load_and_process_image(uploaded_file, max_dim=512):
    img = Image.open(uploaded_file).convert('RGB')
    img.thumbnail((max_dim, max_dim))  # maintain aspect ratio
    img = np.array(img).astype(np.float32) / 255.0
    img = tf.expand_dims(img, axis=0)  # shape: [1, h, w, 3]
    return img

# Convert tensor to PIL image
def tensor_to_image(tensor):
    tensor = tensor * 255
    tensor = tf.cast(tensor[0], tf.uint8)
    return Image.fromarray(tensor.numpy())

# Sidebar: Uploads and blending slider
st.sidebar.header("Upload Images")
content_file = st.sidebar.file_uploader("📷 Content Image", type=["jpg", "jpeg", "png"])
style_file = st.sidebar.file_uploader("🖼️ Style Image", type=["jpg", "jpeg", "png"])
blend = st.sidebar.slider("🎛️ Style Strength", 0.0, 1.0, 1.0, 0.01)
st.sidebar.markdown("Built by Rohith")
if content_file and style_file:
    # Load and process input images
    content_image = load_and_process_image(content_file)
    style_image = load_and_process_image(style_file)

    # Apply style transfer
    stylized_image = model(tf.constant(content_image), tf.constant(style_image))[0]

    # Resize stylized image to match content image shape
    stylized_image = tf.image.resize(stylized_image, size=tf.shape(content_image)[1:3])

    # Blend based on style strength slider
    blended_tensor = (1 - blend) * content_image + blend * stylized_image
    output_image = tensor_to_image(blended_tensor)

    # Convert content and style images to PIL for display
    content_disp = tensor_to_image(content_image)
    style_disp = tensor_to_image(style_image)

    # Layout columns
    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("📷 Content")
        st.image(content_disp, use_container_width=True)
    with col2:
        st.subheader("🖼️ Style")
        st.image(style_disp, use_container_width=True)
    with col3:
        st.subheader("🎨 Stylized")
        st.image(output_image, use_container_width=True)

    # Download button
    buf = io.BytesIO()
    output_image.save(buf, format="PNG")
    st.download_button(
        label="💾 Download Stylized Image",
        data=buf.getvalue(),
        file_name="stylized_output.png",
        mime="image/png"
    )
else:
    st.info("⬅️ Upload both a content image and a style image to get started.")
