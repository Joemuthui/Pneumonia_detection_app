import streamlit as st
import torch
from torchvision import transforms, models
from PIL import Image
import numpy as np
import cv2
import skimage.filters
from custom_model import Custom_Resnet

# Load the model
resnet_50 = models.resnet50(weights=None)
model = Custom_Resnet(resnet_50)
model.load_state_dict(torch.load(r'C:\Users\jmwacira\Documents\Extra_projects\Pneumonia_web\Pneumonia_Detection\model_repo\resnet50_trial_1.pth', map_location='cpu'))
model.eval()

# --- Styling ---
st.set_page_config(page_title="Pneumonia Detection", layout="centered")

st.markdown("""
    <style>
    .main {
        background-color: #f7f9fc;
    }
    .title {
        text-align: center;
        color: #1f77b4;
    }
    .confidence-bar {
        height: 20px;
        border-radius: 10px;
        margin-top: 5px;
    }
    </style>
""", unsafe_allow_html=True)

# --- Grad-CAM ---
def generate_gradcam(model, image_t):
    image_t.requires_grad_()
    outputs = model(image_t)
    target = outputs.max()
    grads = torch.autograd.grad(target, image_t)[0][0][0]
    blurred = skimage.filters.gaussian(grads.detach().cpu().numpy() ** 2, sigma=(15, 15), truncate=4)
    return blurred

# --- Preprocessing ---
def preprocess_image(image):
    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    return val_transforms(image)

# --- App Main ---
def main():
    st.markdown("<h1 class='title'>🩺 Pneumonia Detection from Chest X-ray</h1>", unsafe_allow_html=True)
    st.write("Upload a chest X-ray image to detect pneumonia caused by consolidation or infiltration.")

    uploaded_file = st.file_uploader("📤 Upload X-ray Image", type=["png", "jpg", "jpeg"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')
        image_tensor = preprocess_image(image).unsqueeze(0)

        # --- Prediction ---
        with st.spinner("🧠 Running model prediction..."):
            output = model(image_tensor)
            _, predicted = torch.max(output, 1)
            prediction_label = ['Normal', 'Consolidation', 'Infiltration'][predicted.item()]
            confidence = 100 * output[0][predicted.item()]

        st.markdown(f"### 🧾 **Prediction**: {prediction_label}")
        st.markdown(f"**Confidence:** {confidence:.2f}%")
        # st.progress(confidence / 100)

        # --- Grad-CAM ---
        with st.spinner("🔍 Generating Grad-CAM..."):
            gradcam = generate_gradcam(model, image_tensor)
            gradcam_normalized = (gradcam - gradcam.min()) / (gradcam.max() - gradcam.min())
            heatmap = cv2.applyColorMap(np.uint8(255 * gradcam_normalized), cv2.COLORMAP_JET)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

            image_np = np.array(image.resize((224, 224)))
            overlay = (0.3 * heatmap + image_np).astype(np.uint8)

        # --- Layout with columns ---
        col1, col2 = st.columns(2)
        with col1:
            st.image(image_np, caption="Original Chest X-ray", use_container_width=True)
        with col2:
            st.image(overlay, caption="🧠 Grad-CAM: Important Regions for Prediction", use_container_width=True)

        st.markdown("---")
        st.info("**Note**: This model is a prototype and should not be used for clinical decision-making.")

if __name__ == "__main__":
    main()

