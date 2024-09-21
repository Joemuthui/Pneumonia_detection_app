import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from custom_model import Custom_Resnet
import skimage
import cv2
from torchvision import models

resnet_50=models.resnet50(weights=None)
model = Custom_Resnet(resnet_50)

model.load_state_dict(torch.load('model_repo/resnet50_trial_1.pth', map_location='cpu'))
model.eval()

page_bg="""
<style>
{
primaryColor='#F63366'
backgroundColor='#FFFFFF'
secondaryBackgroundColor='#F0F2F6'
textColor='#262730'
font='sans serif'}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)
# Grad-CAM utilities
def generate_gradcam(model, image_t):
    img = image_t.requires_grad_()


    outputs = model(img)
    target=outputs.max()
    grads = torch.autograd.grad(target, img)[0][0][0]
    blurred = skimage.filters.gaussian(grads.detach().cpu().numpy()**2, sigma=(15, 15), truncate=4)

    return blurred

# Image preprocessing function
def preprocess_image(image):
    val_transforms = transforms.Compose([
        #transforms.Grayscale(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    return val_transforms(image)

# Streamlit app
def main():
    st.title("Pneumonia Detection Based on Consolidation or Infiltration")

    st.write("Upload a chest X-ray image to predict if the patient has pneumonia.")

    uploaded_file = st.file_uploader("Choose an X-ray image...", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file).convert('RGB')
        #st.image(image, caption="Uploaded X-ray", use_column_width=True)
        
        # Preprocess the image
        image_tensor = preprocess_image(image).unsqueeze(0)  # Add batch dimension
        
        # Get model prediction
        with st.spinner('Running model prediction...'):
            output = model(image_tensor)
            soft=output.data
            _, predicted = torch.max(output.data, 1)
            if predicted.item() == 0:
                prediction='Normal'
            elif predicted.item() == 1:
                prediction='Consolidation'
            else:
                prediction='Infiltration'

            st.subheader(f"**Prediction:** {prediction} with  {100*soft[0][predicted.item()]:.2f}% confidence")
        
        # Generate Grad-CAM
        with st.spinner('Generating Grad-CAM...'):
            gradcam = generate_gradcam(model, image_tensor)
            gradcam_normalized = (gradcam - gradcam.min()) / (gradcam.max() - gradcam.min())
            heatmap = cv2.applyColorMap(np.uint8(255 * gradcam_normalized), cv2.COLORMAP_JET)
            
            # Convert to RGB format
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            # Convert PIL image to numpy array
            image_np = np.array(image.resize((224, 224)))
            # Combine heatmap with the original image
            overlay = (0.2 * heatmap + image_np).astype(np.uint8)
            # Display Grad-CAM overlay
            # Create columns for side-by-side display
            col1, col2 = st.columns(2)
            with col1:
                st.image(image_np, caption="Uploaded X-ray", use_column_width=True)
            
            with col2:
                st.image(overlay, caption="Grad-CAM Overlay: The highlighted regions indicates where the model found important at ditermining the prediction", use_column_width=True)


if __name__ == "__main__":
    main()
