# ChronoFace - Age Prediction & Aging Simulation

ChronoFace is an interactive Streamlit-based application that uses Deep Learning models to estimate a person's age from a photo and simulate their aging or rejuvenation.

This project uses:
- **Swin Transformer** (DEX) for precise age prediction.
- **CycleGAN** for facial transformations (Aging & Rejuvenation).

## 🚀 Installation

### 1. Clone the project
Download or clone this repository to your local machine.

### 2. Create a virtual environment (Recommended)
```bash
python -m venv .venv
source .venv/bin/activate  # On Mac/Linux
# or
.venv\Scripts\activate     # On Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

## 📥 Downloading Model Weights (Mandatory)

For the application to work, you must download the following **3 weight files (.pth)** and place them in the project root:

1. `checkpoint_fold_3.pth`: Age prediction model (Swin Transformer).
2. `best_G_AB.pth`: Aging model (CycleGAN).
3. `G_BA_45.pth`: Rejuvenation model (CycleGAN).

👉 **You can download these files here:** [Google Drive Link](https://drive.google.com/drive/folders/10_EO8AE7Fqk7aJcVDLWeXDiSA9fClN15?usp=drive_link)

*Once downloaded, make sure they are in the same folder as the `app.py` file.*

## 🎮 Launch the Application

Once the dependencies are installed and the weight files are downloaded, launch the interface with the following command:

```bash
streamlit run app.py
```

The application will automatically open in your default browser (usually at `http://localhost:8501`).

## 👥 Authors
- **Ouriel Mimoun**
- **David Ceylon**

---
*Academic project built with Streamlit and PyTorch.*
