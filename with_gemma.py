import streamlit as st
import torch
from torchvision import transforms, models
from PIL import Image
import numpy as np
import cv2
import skimage.filters
from transformers import pipeline, BitsAndBytesConfig
from huggingface_hub import login
import logging
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
class Config:
    MODEL_PATH = 'resnet50_trial_1.pth'
    LOGO_PATH = 'pneusight_logo.png'
    IMAGE_SIZE = (224, 224)
    CLASS_LABELS = ['Normal', 'Consolidation', 'Infiltration']
    COLOR_MAP = {
        "Normal": "#28a745",
        "Consolidation": "#ffc107", 
        "Infiltration": "#dc3545"
    }
    MEDGEMMA_MODEL_ID = "google/medgemma-4b-it"
    MAX_FILE_SIZE_MB = 10
    ALLOWED_EXTENSIONS = ["png", "jpg", "jpeg"]

# Custom CSS styling
CUSTOM_CSS = """
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
    margin-bottom: 20px;
}

/* Buttons */
.stButton > button {
    background-color: #262730;
    color: #FAFAFA;
    border: 1px solid #FAFAFA;
    border-radius: 5px;
}

.stButton > button:hover {
    background-color: #3d3d3d;
}

/* File uploader */
.stFileUploader {
    background-color: #1e1e1e !important;
}

/* Result cards */
.result-card {
    border: 1px solid #444;
    border-radius: 8px;
    padding: 15px;
    text-align: center;
    margin: 5px;
    background-color: #1a1a1a;
}

/* Error messages */
.error-message {
    background-color: #2d1b1b;
    border: 1px solid #dc3545;
    border-radius: 5px;
    padding: 10px;
    color: #dc3545;
}

/* Success messages */
.success-message {
    background-color: #1b2d1b;
    border: 1px solid #28a745;
    border-radius: 5px;
    padding: 10px;
    color: #28a745;
}
</style>
"""

# LLM System Prompt
SYSTEM_PROMPT = """
You are a medical imaging expert AI trained to analyze chest X-rays and identify pneumonia patterns.
Given a chest X-ray image, you must determine whether the scan shows NORMAL, INFILTRATION, or CONSOLIDATION patterns based on radiological signs such as:

- Consolidation: Dense, homogeneous opacifications with air bronchograms
- Infiltration: Patchy, heterogeneous opacities with interstitial patterns
- Normal: Clear lung fields with normal vascular markings

Always include a step-by-step explanation based on observed patterns in the image, and mention the level of confidence (e.g., high, moderate, low).
Be cautious not to make a definitive diagnosis but rather offer a likelihood-based reasoning for educational purposes.
"""

USER_QUERY = """Based on the attached chest X-ray, can you analyze and tell me whether the patient has a Normal, Infiltration, or Consolidation pattern? Please explain your reasoning step by step."""

class ModelManager:
    """Handles model loading and caching"""
    
    @staticmethod
    @st.cache_resource
    def load_resnet_model():
        """Load and cache the custom ResNet model"""
        try:
            if not Path(Config.MODEL_PATH).exists():
                raise FileNotFoundError(f"Model file not found: {Config.MODEL_PATH}")
            
            # Import here to avoid circular imports
            from custom_model import Custom_Resnet
            
            resnet_50 = models.resnet50(weights=None)
            model = Custom_Resnet(resnet_50)
            model.load_state_dict(torch.load(Config.MODEL_PATH, map_location='cpu'))
            model.eval()
            logger.info("ResNet model loaded successfully")
            return model
        except Exception as e:
            logger.error(f"Error loading ResNet model: {e}")
            raise

    @staticmethod
    @st.cache_resource
    def load_medgemma_pipeline():
        """Load and cache the MedGemma pipeline"""
        try:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype="float16",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            
            model_kwargs = {
                'torch_dtype': torch.bfloat16,
                'device_map': "auto" if torch.cuda.is_available() else "cpu",
                'quantization_config': quantization_config
            }
            
            pipe = pipeline("image-text-to-text", 
                          model=Config.MEDGEMMA_MODEL_ID, 
                          model_kwargs=model_kwargs)
            pipe.model.generation_config.do_sample = False
            logger.info("MedGemma pipeline loaded successfully")
            return pipe
        except Exception as e:
            logger.error(f"Error loading MedGemma pipeline: {e}")
            raise

class ImageProcessor:
    """Handles image preprocessing and validation"""
    
    @staticmethod
    def validate_image(uploaded_file) -> Tuple[bool, str]:
        """Validate uploaded image file"""
        if uploaded_file is None:
            return False, "No file uploaded"
        
        # Check file size
        if uploaded_file.size > Config.MAX_FILE_SIZE_MB * 1024 * 1024:
            return False, f"File size exceeds {Config.MAX_FILE_SIZE_MB}MB limit"
        
        # Check file extension
        file_extension = uploaded_file.name.split('.')[-1].lower()
        if file_extension not in Config.ALLOWED_EXTENSIONS:
            return False, f"Invalid file type. Allowed: {', '.join(Config.ALLOWED_EXTENSIONS)}"
        
        return True, "Valid file"

    @staticmethod
    def preprocess_image(image: Image.Image) -> torch.Tensor:
        """Preprocess image for model inference"""
        transform = transforms.Compose([
            transforms.Resize(Config.IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])  # ImageNet normalization
        ])
        return transform(image).unsqueeze(0)

class GradCAMGenerator:
    """Generates Grad-CAM visualizations"""
    
    @staticmethod
    def generate_gradcam(model: torch.nn.Module, image_tensor: torch.Tensor) -> np.ndarray:
        """Generate Grad-CAM heatmap"""
        try:
            image_tensor.requires_grad_()
            outputs = model(image_tensor)
            target = outputs.max()
            
            # Clear previous gradients
            model.zero_grad()
            
            # Compute gradients
            target.backward()
            gradients = image_tensor.grad[0][0].detach().cpu().numpy()
            
            # Apply Gaussian blur
            heatmap = skimage.filters.gaussian(gradients ** 2, sigma=(15, 15), truncate=4)
            return heatmap
        except Exception as e:
            logger.error(f"Error generating Grad-CAM: {e}")
            raise

    @staticmethod
    def create_overlay(original_image: np.ndarray, heatmap: np.ndarray) -> np.ndarray:
        """Create overlay of original image and heatmap"""
        # Normalize heatmap
        heatmap_normalized = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
        
        # Apply colormap
        colored_heatmap = cv2.applyColorMap(np.uint8(255 * heatmap_normalized), cv2.COLORMAP_JET)
        colored_heatmap = cv2.cvtColor(colored_heatmap, cv2.COLOR_BGR2RGB)
        
        # Create overlay
        overlay = (0.3 * colored_heatmap + 0.7 * original_image).astype(np.uint8)
        return overlay

class PneumoniaDetector:
    """Main pneumonia detection class"""
    
    def __init__(self):
        self.resnet_model = None
        self.medgemma_pipeline = None
        self.load_models()

    def load_models(self):
        """Load all required models"""
        try:
            self.resnet_model = ModelManager.load_resnet_model()
        except Exception as e:
            st.error(f"❌ Failed to load ResNet model: {e}")
            return
        
        try:
            self.medgemma_pipeline = ModelManager.load_medgemma_pipeline()
        except Exception as e:
            st.warning("⚠️ MedGemma model not available. AI analysis will be disabled.")
            self.medgemma_pipeline = None

    def predict(self, image_tensor: torch.Tensor) -> Tuple[str, float]:
        """Make prediction using ResNet model"""
        with torch.no_grad():
            outputs = self.resnet_model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            predicted_idx = torch.argmax(probabilities).item()
            confidence = probabilities[predicted_idx].item() * 100
            
            prediction_label = Config.CLASS_LABELS[predicted_idx]
            return prediction_label, confidence

    def get_ai_analysis(self, image: Image.Image, api_token: str) -> Optional[str]:
        """Get AI analysis from MedGemma model"""
        if not self.medgemma_pipeline or not api_token:
            return None
        
        try:
            login(token=api_token)
            messages = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": SYSTEM_PROMPT}]
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": USER_QUERY},
                        {"type": "image", "image": image}
                    ]
                }
            ]
            
            output = self.medgemma_pipeline(text=messages, max_new_tokens=2048)
            return output[0]["generated_text"][-1]["content"]
        except Exception as e:
            logger.error(f"Error in AI analysis: {e}")
            return f"Error in AI analysis: {str(e)}"

def render_ui():
    """Render the main UI"""
    st.set_page_config(
        page_title="PneuSight | Pneumonia Detection", 
        layout="centered",
        initial_sidebar_state="expanded"
    )
    
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    
    # Header
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if Path(Config.LOGO_PATH).exists():
            st.image(Config.LOGO_PATH, width=200)
    
    st.markdown("<h1 class='title'>🩺 AI-Powered Pneumonia Detection</h1>", unsafe_allow_html=True)

def render_results(prediction: str, confidence: float):
    """Render prediction results"""
    pred_color = Config.COLOR_MAP.get(prediction, "#17a2b8")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(
            f"""
            <div class="result-card">
                <h4>🧾 Prediction</h4>
                <p style="color: {pred_color}; font-size: 24px; font-weight: bold;">{prediction}</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    with col2:
        st.markdown(
            f"""
            <div class="result-card">
                <h4>📊 Confidence</h4>
                <p style="font-size: 24px; font-weight: bold;">{confidence:.1f}%</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    with col3:
        st.markdown(
            """
            <div class="result-card">
                <h4>📈 Score</h4>
            """,
            unsafe_allow_html=True
        )
        st.progress(confidence / 100)
        st.markdown("</div>", unsafe_allow_html=True)

def main():
    """Main application function"""
    render_ui()
    
    # Initialize detector
    if 'detector' not in st.session_state:
        with st.spinner("🔄 Loading models..."):
            st.session_state.detector = PneumoniaDetector()
    
    detector = st.session_state.detector
    
    if detector.resnet_model is None:
        st.error("❌ Cannot proceed without the main detection model.")
        return
    
    # Sidebar for API token
    with st.sidebar:
        st.header("Configuration")
        api_token = st.text_input(
            "🔑 HuggingFace API Token (Optional)", 
            type="password",
            help="Required for AI analysis feature"
        )
        
        st.markdown("---")
        st.markdown("### About")
        st.info("This app detects pneumonia patterns in chest X-rays using deep learning.")
        
        st.markdown("### Model Info")
        st.write("- **Detection**: Custom ResNet-50")
        st.write("- **Analysis**: Google MedGemma-4B")
    
    # File upload
    uploaded_file = st.file_uploader(
        "📤 Upload Chest X-ray Image", 
        type=Config.ALLOWED_EXTENSIONS,
        help=f"Supported formats: {', '.join(Config.ALLOWED_EXTENSIONS)}, Max size: {Config.MAX_FILE_SIZE_MB}MB"
    )
    
    if uploaded_file:
        # Validate file
        is_valid, message = ImageProcessor.validate_image(uploaded_file)
        if not is_valid:
            st.error(f"❌ {message}")
            return
        
        try:
            # Load and process image
            image = Image.open(uploaded_file).convert('RGB')
            image_tensor = ImageProcessor.preprocess_image(image)
            
            # Make prediction
            with st.spinner("🧠 Analyzing image..."):
                prediction, confidence = detector.predict(image_tensor)
            
            # Display results
            render_results(prediction, confidence)
            
            # Generate Grad-CAM
            with st.spinner("🔍 Generating attention map..."):
                try:
                    # Use original image for Grad-CAM (without normalization)
                    image_for_gradcam = transforms.Compose([
                        transforms.Resize(Config.IMAGE_SIZE),
                        transforms.ToTensor(),
                    ])(image).unsqueeze(0)
                    
                    gradcam = GradCAMGenerator.generate_gradcam(detector.resnet_model, image_for_gradcam)
                    
                    # Prepare images for display
                    image_np = np.array(image.resize(Config.IMAGE_SIZE))
                    overlay = GradCAMGenerator.create_overlay(image_np, gradcam)
                    
                    # Display images
                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(image_np, caption="📷 Original X-ray", use_column_width=True)
                    with col2:
                        st.image(overlay, caption="🧠 AI Attention Map", use_column_width=True)
                        
                except Exception as e:
                    st.warning(f"⚠️ Could not generate attention map: {e}")
            
            # AI Analysis
            if detector.medgemma_pipeline and api_token:
                with st.expander("🤖 AI Medical Analysis", expanded=False):
                    with st.spinner("🔬 Generating detailed analysis..."):
                        analysis = detector.get_ai_analysis(image, api_token)
                        if analysis:
                            st.markdown(analysis)
                        else:
                            st.warning("Could not generate analysis. Please check your API token.")
            elif not api_token:
                st.info("💡 **Tip**: Add your HuggingFace API token in the sidebar for detailed AI analysis!")
            
        except Exception as e:
            st.error(f"❌ Error processing image: {e}")
            logger.error(f"Image processing error: {traceback.format_exc()}")
    
    # Footer
    st.markdown("---")
    st.warning(
        "⚠️ **Medical Disclaimer**: This is a prototype for educational purposes only. "
        "It should never be used for clinical decision-making or replace professional medical diagnosis."
    )
    
    # Additional info in expander
    with st.expander("ℹ️ Technical Information"):
        st.markdown("""
        **Model Architecture:**
        - Primary classifier: Custom ResNet-50 trained on chest X-ray data
        - Explainability: Grad-CAM attention visualization
        - Analysis: Google MedGemma-4B multimodal language model
        
        **Classes:**
        - **Normal**: Healthy lung tissue
        - **Consolidation**: Dense opacification (often bacterial pneumonia)
        - **Infiltration**: Patchy inflammation (often viral/atypical pneumonia)
        
        **Performance Notes:**
        - Model performance depends on image quality and positioning
        - Best results with standard PA (posterior-anterior) chest X-rays
        - Lateral views and poor quality images may reduce accuracy
        """)

if __name__ == "__main__":
    main()
