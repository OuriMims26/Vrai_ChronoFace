import streamlit as st
import torch
import torch.nn.functional as F
import torch.nn as nn
import timm
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import os

# --- CUSTOM CSS FOR STYLING ---
st.set_page_config(
    page_title="ChronoFace - Age Prediction & Aging",
    page_icon="⏳",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .big-font {
        font-size:30px !important;
        font-weight: bold;
        color: #FF4B4B;
    }
    .stButton>button {
        width: 100%;
        background-color: #FF4B4B;
        color: white;
        height: 3em;
        font-size: 20px;
    }
    .chrono-title {
        background: linear-gradient(45deg, #FF4B4B, #833ab4, #fd1d1d);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 60px !important;
        font-weight: 900;
        text-align: center;
        margin-bottom: 30px;
        font-family: 'Outfit', sans-serif;
        cursor: pointer;
        transition: transform 0.3s ease;
    }
    .chrono-title:hover {
        transform: scale(1.02);
    }
    </style>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@900&display=swap" rel="stylesheet">
    """, unsafe_allow_html=True)

# --- HEIC SUPPORT ---
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:
    pass

# --- MODEL CONFIGURATION ---
AGE_MODEL_PATH = "checkpoint_fold_3.pth"
GAN_MODEL_PATH = "best_G_AB.pth"
NUM_CLASSES = 121 # 0 à 120 ans (basé sur le checkpoint)

# -------------------------------------------------------------------------------------------------
# ------------------------------ ARCHITECTURES ----------------------------------------------------
# -------------------------------------------------------------------------------------------------

# --- 1. AGE PREDICTION MODEL (Swin Transformer) ---
class SwinAgeModel(nn.Module):
    """Architecture custom correspondant au checkpoint entraîné."""
    def __init__(self, num_classes=121):
        super().__init__()
        # Le backbone est un Swin Transformer
        self.backbone = timm.create_model('swin_base_patch4_window7_224', pretrained=False, num_classes=0)
        
        # Récupérer la taille des features du backbone
        num_features = self.backbone.num_features  # 1024 pour swin_base
        
        # Le head personnalisé
        self.age_head = nn.Sequential(
            nn.Linear(num_features, 512),  
            nn.ReLU(inplace=True),          
            nn.Dropout(0.5),                
            nn.Linear(512, num_classes)     
        )
    
    def forward(self, x):
        features = self.backbone(x)
        return self.age_head(features)

# --- 2. CYCLEGAN GENERATOR (ResNet) ---
class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
        )

    def forward(self, x):
        return x + self.block(x)

class GeneratorResNet(nn.Module):
    def __init__(self, input_shape, num_residual_blocks=9):
        super(GeneratorResNet, self).__init__()

        channels = input_shape[0]

        # Initial convolution block
        out_features = 64
        model = [
            nn.ReflectionPad2d(channels),
            nn.Conv2d(channels, out_features, 7),
            nn.InstanceNorm2d(out_features),
            nn.ReLU(inplace=True),
        ]
        in_features = out_features

        # Downsampling
        for _ in range(2):
            out_features *= 2
            model += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features

        # Residual blocks
        for _ in range(num_residual_blocks):
            model += [ResidualBlock(out_features)]

        # Upsampling
        for _ in range(2):
            out_features //= 2
            model += [
                nn.Upsample(scale_factor=2),
                nn.Conv2d(in_features, out_features, 3, stride=1, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features

        # Output layer
        model += [nn.ReflectionPad2d(channels), nn.Conv2d(out_features, channels, 7), nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

# -------------------------------------------------------------------------------------------------
# ------------------------------ LOADING FUNCTIONS ------------------------------------------------
# -------------------------------------------------------------------------------------------------

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=100)
    st.markdown('<a href="/" target="_self" style="text-decoration:none;"><h2 style="color:white;">⏳ ChronoFace</h2></a>', unsafe_allow_html=True)
    st.subheader("Team Members")
    st.markdown("""
    - **Ouriel Mimoun**
    - **David Ceylon**
    """)
    st.divider()
    show_details = st.checkbox("Show technical details", value=False)

# --- HELPER: GET DEVICE ---
def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

# --- CONFIGURATION ---
torch.set_num_threads(1) # Évite la saturation CPU sur Mac

@st.cache_resource
def load_age_model_v3():
    """Charge le modèle de prédiction d'âge."""
    device = get_device()
    
    try:
        model = SwinAgeModel(num_classes=NUM_CLASSES)
    except Exception as e:
        st.error(f"Age Arch. Error: {e}")
        return None, device

    if os.path.exists(AGE_MODEL_PATH):
        try:
            # map_location needs to be adapted
            checkpoint = torch.load(AGE_MODEL_PATH, map_location=device, weights_only=False)
            if 'model_state_dict' in checkpoint: state_dict = checkpoint['model_state_dict']
            elif 'model' in checkpoint: state_dict = checkpoint['model']
            elif 'state_dict' in checkpoint: state_dict = checkpoint['state_dict']
            else: state_dict = checkpoint
            
            model.load_state_dict(state_dict)
            model.to(device)
            model.eval()
            return model, device
        except Exception as e:
            st.error(f"Age Weights Error: {e}")
            return None, device
    else:
        st.error(f"Age file missing: {AGE_MODEL_PATH}")
        model.to(device)
        model.eval()
        return model, device

@st.cache_resource
def load_gan_model(model_path):
    """Charge le générateur CycleGAN (CPU uniquement pour éviter les crashs MPS)."""
    # Forcer CPU pour le GAN car MPS peut crasher avec certaines opérations
    device = torch.device("cpu")
    
    try:
        # Standard CycleGAN config: (3, 256, 256) and 9 residual blocks
        model = GeneratorResNet(input_shape=(3, 256, 256), num_residual_blocks=9)
    except Exception as e:
        st.error(f"GAN Arch. Error: {e}")
        return None, device

    if os.path.exists(model_path):
        try:
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
            # Basic dict handling
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif isinstance(checkpoint, dict) and 'model' in checkpoint:
                 state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
                
            model.load_state_dict(state_dict)
            model.to(device)
            model.eval()
            return model, device
        except Exception as e:
            st.error(f"GAN Weights Error: {e}")
            return None, device
    else:
        st.error(f"GAN file missing: {model_path}")
        model.to(device)
        model.eval()
        return model, device

# Chargement initial seulement du modèle d'âge pour ne pas surcharger la mémoire
# Chargement différé du modèle d'âge (au clic)
age_model = None
device_age = None
# gan_model chargé à la demande ou en arrière plan

# -------------------------------------------------------------------------------------------------
# ------------------------------ INTERFACE & LOGIC ------------------------------------------------
# -------------------------------------------------------------------------------------------------

st.markdown('<a href="/" target="_self" style="text-decoration:none;"><h1 class="chrono-title">ChronoFace</h1></a>', unsafe_allow_html=True)
st.markdown("### Deep EXpectation (DEX) + CycleGAN")
st.write(f"Device used: `{device_age}`")

st.divider()

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("1. Upload an image")
    uploaded_file = st.file_uploader("Drag and drop your image here", type=['jpg', 'jpeg', 'png', 'heic'])

    image = None
    if uploaded_file is not None:
        try:
            uploaded_file.seek(0)
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption="Image loaded", use_column_width=True)
        except Exception as e:
            uploaded_file.seek(0)
            header = uploaded_file.read(12) # Read more for better header info
            st.error(f"Error: Unable to read image. This may be due to an unsupported format.")
            st.code(f"Header (Hex): {header.hex()}\nError: {e}", language="text")
            if "heic" in header.hex().lower() or "hea1" in header.hex().lower():
                st.info("💡 It looks like you're trying to upload an HEIC/HEIF image. HEIC support has just been added, please try reloading the app.")

with col2:
    st.subheader("2. Analysis & Aging")
    if image is not None:
        # Initialize session state for persistence
        if "age_result" not in st.session_state:
            st.session_state.age_result = None
        if "gan_result" not in st.session_state:
            st.session_state.gan_result = None
        
        # Reset state if new file uploaded
        if "current_file_name" not in st.session_state or st.session_state.current_file_name != uploaded_file.name:
            st.session_state.age_result = None
            st.session_state.gan_result = None
            st.session_state.current_file_name = uploaded_file.name

        # --- STEP 1: AGE PREDICTION ---
        if st.button("1. Predict Age 🚀"):
            if age_model is None:
                # Load age model on demand
                age_model, device_age = load_age_model_v3()
                if age_model is None:
                    st.error("Unable to load age model.")
                    st.stop()
            # Model is now loaded
            with st.spinner('Estimating age...'):
                try:
                    transform_age = transforms.Compose([
                        transforms.Resize((224, 224)),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    ])
                    input_tensor = transform_age(image).unsqueeze(0).to(device_age)
                    
                    with torch.no_grad():
                        output = age_model(input_tensor)
                        probs = F.softmax(output, dim=1)
                        age_range = torch.arange(NUM_CLASSES).float().to(device_age)
                        st.session_state.age_result = {
                            "age": (probs * age_range).sum(dim=1).item(),
                            "top_probs": torch.topk(probs, 5)[0].cpu().numpy().flatten(),
                            "top_indices": torch.topk(probs, 5)[1].cpu().numpy().flatten()
                        }
                    st.success("Age predicted!")
                except Exception as e:
                    st.error(f"Error predicting age: {e}")

        # DISPLAY AGE IF EXISTS
        if st.session_state.age_result:
            res = st.session_state.age_result
            st.markdown(f'<p class="big-font">Estimated Age: {res["age"]:.1f} years</p>', unsafe_allow_html=True)
            
            with st.expander("📊 Show Prediction Distribution"):
                fig, ax = plt.subplots(figsize=(6, 3))
                ax.bar(res["top_indices"], res["top_probs"], color='#FF4B4B', alpha=0.7)
                ax.set_xlabel("Age")
                st.pyplot(fig)
            
            if show_details:
                st.json({"Age": res["age"], "Top1": int(res["top_indices"][0])})

        st.divider()

        # --- GAN AGING BLOCK ---
        st.markdown("### 👴 Aging Simulation")
        
        # Transformation type selection
        gan_option = st.selectbox("Choose desired effect", 
                                 ["👴 Aging", "👶 Rejuvenation"],
                                 index=0)
        
        selected_model_path = "best_G_AB.pth" if "Aging" in gan_option else "G_BA_45.pth"
        
        if st.button("2. Transform ✨"):
            selected_model_path = "best_G_AB.pth" if "Aging" in gan_option else "G_BA_45.pth"
            gan_model, device_gan = load_gan_model(selected_model_path)
            
            if gan_model is None:
                st.error(f"GAN Model not loaded ({selected_model_path} missing?)")
            else:
                try:
                    with st.spinner(f'Running {gan_option}...'):
                        transform_gan = transforms.Compose([
                            transforms.Resize((256, 256)),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                        ])
                        input_gan = transform_gan(image).unsqueeze(0).to(device_gan)
                        
                        with torch.no_grad():
                            fake_old = gan_model(input_gan)
                            fake_old = fake_old * 0.5 + 0.5
                            fake_old = torch.clamp(fake_old, 0, 1)
                            st.session_state.gan_result = {
                                "image": fake_old.squeeze(0).permute(1, 2, 0).cpu().numpy(),
                                "label": gan_option
                            }
                        st.success("Transformation complete!")
                except Exception as e:
                    st.error(f"Error during transformation: {e}")

        # DISPLAY GAN IF EXISTS
        if st.session_state.gan_result:
            res_gan = st.session_state.gan_result
            res_col1, res_col2 = st.columns(2)
            img_resized = image.resize((256, 256))
            with res_col1:
                st.image(img_resized, caption="Original", use_column_width=True)
            with res_col2:
                st.image(res_gan["image"], caption=f"ChronoFace: {res_gan['label']}", use_column_width=True)

# --- FOOTER ---
st.divider()
st.markdown("Academic project made with Streamlit, PyTorch (Swin Transformer & CycleGAN).")
