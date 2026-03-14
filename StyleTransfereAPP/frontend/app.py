import streamlit as st
import requests
from PIL import Image
import io
import base64

st.set_page_config(
    page_title="Neural Style Transfer",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🎨 Neural Style Transfer")

# Upload images in sidebar for cleaner layout
st.sidebar.header("Upload Images")
content = st.sidebar.file_uploader("Content Image", type=["png", "jpg", "jpeg"])
style = st.sidebar.file_uploader("Style Image", type=["png", "jpg", "jpeg"])

if content and style:
    # Display the uploaded images
    st.subheader("Uploaded Images")
    col1, col2 = st.columns(2)
    with col1:
        st.caption("Content Image")
        st.image(content, use_column_width=True)
    with col2:
        st.caption("Style Image")
        st.image(style, use_column_width=True)

    if st.button("🎨 Transfer Style"):
        # Prepare files
        files = {
            "content": content.getvalue(),
            "style": style.getvalue()
        }

        # Call backend API
        with st.spinner("Applying style transfer... This may take a few minutes ⏳"):
            response = requests.post(
                "http://127.0.0.1:8000/style-transfer",
                files={
                    "content": ("content.png", files["content"]),
                    "style": ("style.png", files["style"])
                }
            )

        if response.status_code == 200:
            data = response.json()
            if "image" in data:
                img_bytes = base64.b64decode(data["image"])
                img = Image.open(io.BytesIO(img_bytes))
                st.subheader("Result")
                st.image(img, use_column_width=True)
            else:
                st.error("Error from backend: " + data.get("error", "Unknown error"))
        else:
            st.error(f"Request failed with status {response.status_code}")
else:
    st.info("Please upload both a content image and a style image from the sidebar.")