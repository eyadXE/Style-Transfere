# 🎨 Neural Style Transfer Web App

> Transform any image into a work of art by blending content and style using deep learning.

---

## 📖 Project Description

This project implements a **Neural Style Transfer (NST)** web application that allows users to apply the artistic style of one image *(style image)* to another image *(content image)*. The project leverages:

- **PyTorch** — for neural network computations
- **FastAPI** — for serving the style transfer model
- **Streamlit** — for an interactive, user-friendly frontend

Neural Style Transfer is based on the concept of separating content features and style features using convolutional neural networks (CNNs), then generating a new image that combines the **content** of one image with the **style** of another.

---
### Demo
![Demo](demo1.gif)
## ✨ Features

- 📁 Upload content and style images via a simple sidebar
- 🖼️ Display original images side by side before transfer
- 🧠 Apply style transfer using a pre-trained CNN (AlexNet)
- 🖼️ Display the stylized image in large format
- ⚡ Fully asynchronous FastAPI backend to handle NST computation
- 📦 Handles large images and encodes results in Base64 for efficient transfer

---

## 📁 File Structure

```
StyleTransferApp/
│
├── backend/
│   ├── style_transfer.py       # Core neural style transfer code (PyTorch)
│   ├── model.py                # CNN model and feature map utilities
│   └── api.py                  # FastAPI server exposing /style-transfer endpoint
│
├── frontend/
│   └── app.py                  # Streamlit frontend
│
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

---

## ⚙️ How It Works

### Backend (`backend/`)

#### `model.py`
- Loads the pre-trained CNN (AlexNet)
- Provides utilities for extracting feature maps and computing **Gram matrices** for style representation

#### `style_transfer.py`
Implements the NST algorithm:
1. Extract content and style features (detached to prevent unnecessary backprop)
2. Optimize a target image to minimize **content loss** and **style loss**
3. Returns a stylized image as a PyTorch tensor

#### `api.py`
- FastAPI server with one `POST` endpoint `/style-transfer`
- Accepts uploaded content and style images
- Runs NST and returns the resulting image encoded in **Base64**

---

### Frontend (`frontend/app.py`)

Uses **Streamlit** to provide an interactive UI with:
- Sidebar file upload for content and style images
- Side-by-side display of original images
- Style transfer trigger button with a loading spinner
- Display of the final stylized image

---

## 🚀 Installation

1. **Clone the repository:**

```bash
git clone https://github.com/yourusername/StyleTransferApp.git
cd StyleTransferApp
```

2. **Create a virtual environment and activate it:**

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

3. **Install dependencies:**

```bash
pip install -r requirements.txt
```

4. **Run the backend FastAPI server:**

```bash
uvicorn backend.api:app --reload
```

5. **Run the frontend Streamlit app:**

```bash
streamlit run frontend/app.py
```

6. Open your browser at **`http://localhost:8501`**

---

## 🖱️ Usage

1. Upload a **content image** — the image whose structure you want to preserve
2. Upload a **style image** — the image whose artistic style you want to apply
3. Click **"Transfer Style"**
4. View the **stylized result** alongside the original images

---

## 📚 References

- Gatys, L. A., Ecker, A. S., & Bethge, M. (2016). *A Neural Algorithm of Artistic Style.* Journal of Vision. [Paper Link](https://arxiv.org/abs/1508.06576)
- PyTorch Official Documentation: https://pytorch.org/docs/stable/index.html
- FastAPI Documentation: https://fastapi.tiangolo.com/
- Streamlit Documentation: https://docs.streamlit.io/
- Krizhevsky, A., Sutskever, I., & Hinton, G. (2012). *ImageNet Classification with Deep Convolutional Neural Networks.* [Paper Link](https://papers.nips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html)

---

## 🔮 Notes / Future Improvements

- [ ] Add progress bar for NST optimization epochs
- [ ] Optimize for GPU inference to reduce processing time
- [ ] Add multi-style transfer and blend factor slider
- [ ] Handle very large images with dynamic resizing to avoid memory issues

---

## 📄 License

This project is open source. See [`LICENSE`](./LICENSE) for details.
