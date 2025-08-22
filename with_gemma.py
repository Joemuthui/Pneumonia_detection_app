import streamlit as st
import torch
from torchvision import transforms, models
from PIL import Image
import numpy as np
import cv2
import skimage.filters
from custom_model import Custom_Resnet

# MedGemma imports
from transformers import pipeline, BitsAndBytesConfig
from huggingface_hub import login

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

# --- MedGemma Integration ---
@st.cache_resource
def load_medgemma_pipeline():
    """Load MedGemma pipeline with caching."""
    try:
        use_cuda = torch.cuda.is_available()
        device_map = "auto" if use_cuda else None
        torch_dtype = torch.bfloat16 if use_cuda else torch.float32
        
        model_kwargs = {}
        if use_cuda:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            model_kwargs["quantization_config"] = quant_config
        
        vlm = pipeline(
            # task="image-to-text",
            task = "image-text-to-text",
            model="google/medgemma-4b-it",
            torch_dtype=torch_dtype,
            device_map=device_map,
            model_kwargs=model_kwargs,
        )
        
        if hasattr(vlm.model, "generation_config"):
            vlm.model.generation_config.do_sample = False
        
        return vlm
    except Exception as e:
        st.error(f"Failed to load MedGemma: {e}")
        return None

def analyze_with_medgemma(image, hf_token=None):
    """Analyze image with MedGemma."""
    try:
        # Authenticate if token provided
        if hf_token:
            login(token=hf_token)
        
        # Load pipeline
        vlm = load_medgemma_pipeline()
        print(vlm)
        if vlm is None:
            return None
            
        system_prompt = """You are a medical imaging expert AI trained to analyze chest X-rays and identify pneumonia patterns. Given a chest X-ray image, determine whether the scan shows NORMAL, INFILTRATION, or CONSOLIDATION patterns. Provide a short, step-by-step explanation and a confidence descriptor (high/moderate/low). Do not provide medical advice."""
        
        user_prompt = """Analyze this chest X-ray and state whether it is more consistent with Normal, Infiltration, or Consolidation. Explain briefly."""
        
        # prompt = system_prompt + " " + user_prompt
        
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}]
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "image", "image": image}
                ]
            }
        ]
                
        # result = vlm(image, prompt=prompt, max_new_tokens=512)
        result = vlm(text=messages, max_new_tokens=2048)
        # response = output[0]["generated_text"][-1]["content"]
        
        if isinstance(result, list) and len(result) and isinstance(result[0], dict) and "generated_text" in result[0]:
            return result[0]["generated_text"][-1]["content"]
        return str(result)
        
    except Exception as e:
        return f"MedGemma analysis failed: {e}"

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
        #st.caption("AI-Powered Pneumonia Detection")
        # st.markdown("<h1 style='text-align: center;'>PneuSight</h1>", unsafe_allow_html=True)
    st.markdown("<h1 class='title'>🩺 AI-Powered Pneumonia Detection</h1>", unsafe_allow_html=True)

    # MedGemma Configuration in Sidebar
    with st.sidebar:
        st.subheader("🤖 MedGemma Configuration")
        use_medgemma = st.checkbox("Enable MedGemma Analysis", value=False)
        hf_token = None
        if use_medgemma:
            hf_token = st.text_input("HuggingFace Token", type="password", help="Required for MedGemma access")

    uploaded_file = st.file_uploader("📤 Upload Chest X-ray Image", type=["png", "jpg", "jpeg"])

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
            st.image(image_np, caption="Original Chest X-ray", use_container_width=True)
        with col2:
            st.image(overlay, caption="🧠 Grad-CAM: Important Regions for Prediction", use_container_width=True)

        # --- MedGemma Analysis Section ---
        if use_medgemma:
            st.markdown("---")
            st.subheader("🤖 MedGemma Expert Analysis")
            
            if not hf_token:
                st.warning("⚠️ HuggingFace token required for MedGemma analysis")
            else:
                with st.spinner("🔬 Generating expert medical analysis..."):
                    analysis = analyze_with_medgemma(image, hf_token)
                
                if analysis:
                    st.markdown(
                        f"""
                        <div style="border: 1px solid #444; border-radius: 8px; padding: 20px; background-color: #1a1a1a;">
                            <h5 style="color: #F63366; margin-bottom: 15px;">Medical AI Interpretation:</h5>
                            <p style="color: #FAFAFA; line-height: 1.6;">{analysis}</p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                else:
                    st.error("❌ MedGemma analysis failed. Please check your token and try again.")

    st.markdown("---")
    st.info("**Note**: This model is a prototype and should not be used for clinical decision-making.")

if __name__ == "__main__":
    main()