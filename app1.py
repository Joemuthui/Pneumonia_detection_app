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
model.load_state_dict(torch.load('resnet50_trial_1.pth', map_location='cpu'))
model.eval()

st.set_page_config(page_title="PneuSight | Pneumonia Detection", layout="centered")

st.markdown("""
    <style>
    /* Overall background and text */
    html, body, [class*="css"] {
        background-color: #0E1117 !important;
        color: #FAFAFA !important;
    }

    /* Headings */
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
    }

    /* Centered title */
    .title {
        text-align: center;
        color: #F63366;
        font-size: 36px;
        margin-bottom: 0px;
    }

    /* Buttons */
    .stButton > button {
        background-color: #262730;
        color: #FAFAFA;
        border: 1px solid #FAFAFA;
    }

    /* File uploader */
    .stFileUploader {
        background-color: #1e1e1e !important;
    }

    /* Custom class (you can target specific elements this way) */
    .confidence-bar {
        height: 20px;
        border-radius: 10px;
        margin-top: 5px;
        background-color: #1f77b4;
    }
    </style>
""", unsafe_allow_html=True)
# Determine color based on prediction
color_map = {
    "Normal": "#28a745",         # Green
    "Consolidation": "#ffc107", # Amber
    "Infiltration": "#dc3545"   # Red
}
class_labels = ['Normal', 'Consolidation', 'Infiltration']

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
    col = st.columns(3)[1]  # Center column out of 3

    with col:
        st.image("pneusight_logo.png", width=200)
        st.caption("AI-Powered Pneumonia Detection")
        # st.markdown("<h1 style='text-align: center;'>PneuSight</h1>", unsafe_allow_html=True)
    st.markdown("<h1 class='title'>🩺 Pneumonia Detection from Chest X-ray</h1>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("📤 Upload X-ray Image", type=["png", "jpg", "jpeg"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')
        image_tensor = preprocess_image(image).unsqueeze(0)

        # --- Prediction ---
        with st.spinner("🧠 Running model prediction..."):
            output = model(image_tensor)
            _, predicted = torch.max(output, 1)
            prediction_label = ['Normal', 'Consolidation', 'Infiltration'][predicted.item()]
            confidence_value = float(output[0][predicted.item()].item()) * 100
            
    pred_color = color_map.get(prediction_label, "#17a2b8")  # default = info blue

    # Create 3 styled columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(
            f"""
            <div style="border: 1px solid #444; border-radius: 8px; padding: 15px; text-align: center;">
                <h4 style="color: #ffffff;">🧾 Prediction</h4>
                <p style="color: {pred_color}; font-size: 24px; font-weight: bold;">{prediction_label}</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    with col2:
        st.markdown(
            f"""
            <div style="border: 1px solid #444; border-radius: 8px; padding: 15px; text-align: center;">
                <h4 style="color: #ffffff;">📊 Confidence</h4>
                <p style="font-size: 24px; font-weight: bold;">{confidence_value:.2f}%</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    with col3:
        st.markdown(
            f"""
            <div style="border: 1px solid #444; border-radius: 8px; padding: 15px;">
                <h4 style="text-align: center; color: #ffffff;">📈 Score</h4>
            """,
            unsafe_allow_html=True
        )
        st.progress(confidence_value / 100)

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
        st.image(image_np, caption="Original Chest X-ray", use_column_width=True)
    with col2:
        st.image(overlay, caption="🧠 Grad-CAM: Important Regions for Prediction", use_column_width=True)

    st.markdown("---")
    st.info("**Note**: This model is a prototype and should not be used for clinical decision-making.")

if __name__ == "__main__":
    main()

