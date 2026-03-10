{\rtf1\ansi\ansicpg1252\cocoartf2709
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fnil\fcharset0 HelveticaNeue;}
{\colortbl;\red255\green255\blue255;\red220\green220\blue220;\red30\green31\blue33;}
{\*\expandedcolortbl;;\cssrgb\c89020\c89020\c89020;\cssrgb\c15686\c16471\c17255;}
\paperw11900\paperh16840\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\deftab720
\pard\pardeftab720\partightenfactor0

\f0\fs32 \AppleTypeServices\AppleTypeServicesF65539 \cf2 \cb3 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec2 from google.colab import drive\
import os\
\
# Connexion au Google Drive\
drive.mount('/content/drive')\
\
# Chemin vers ton dossier existant\
PROJECT_PATH = "/content/drive/MyDrive/Projet_IA"\
\
if os.path.exists(PROJECT_PATH):\
print(f"\uc0\u9989  Dossier Projet_IA d\'e9tect\'e9 sur le Drive.")\
else:\
print(f"\uc0\u10060  Erreur : Dossier Projet_IA introuvable. V\'e9rifie le nom sur ton Drive.")\
\
# Installation des biblioth\'e8ques n\'e9cessaires\
!pip install timm tqdm pandas pillow scikit-learn\
\
import torch\
import torch.nn as nn\
import torch.nn.functional as F\
import timm\
import pandas as pd\
from torch.utils.data import Dataset, DataLoader\
from torchvision import transforms\
from PIL import Image\
from sklearn.model_selection import train_test_split\
from tqdm.auto import tqdm\
\
# --- CONFIGURATION DES CHEMINS ---\
ZIP_PATH = "/content/drive/MyDrive/Projet_IA/dataset.zip"\
EXTRACT_PATH = "/content/dataset_local" # Disque rapide temporaire de Colab\
SAVE_PATH = "/content/drive/MyDrive/Projet_IA/best_swin_dex_colab.pth"\
\
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")\
NUM_CLASSES = 101 # 0 \'e0 100 ans\
\
# Extraction du ZIP\
if not os.path.exists(EXTRACT_PATH):\
print("\uc0\u55357 \u56550  Extraction du dataset en cours (sois patient)...")\
!unzip -q "\{ZIP_PATH\}" -d "\{EXTRACT_PATH\}"\
\
# Scan des images et extraction des \'e2ges depuis les noms de fichiers\
file_paths, ages = [], []\
for root, _, files in os.walk(EXTRACT_PATH):\
for filename in files:\
if filename.lower().endswith((".jpg", ".jpeg", ".png")):\
try:\
parts = filename.split('_')\
# Logique : ID_Naissance_Photo.jpg\
annee_photo = int(parts[2].split('-')[0].split('.')[0])\
annee_naissance = int(parts[1].split('-')[0])\
age = annee_photo - annee_naissance\
if 0 <= age <= 100:\
file_paths.append(os.path.join(root, filename))\
ages.append(age)\
except: continue\
\
df = pd.DataFrame(\{'path': file_paths, 'age': ages\})\
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['age'])\
print(f"\uc0\u9989  \{len(df)\} images pr\'eates. Train: \{len(train_df)\} | Val: \{len(val_df)\}")\
\
# Hyperparam\'e8tres\
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)\
criterion = nn.CrossEntropyLoss()\
age_range = torch.arange(NUM_CLASSES).float().to(DEVICE)\
\
# DataLoaders avec transformations de base\
class AgeDataset(Dataset):\
def __init__(self, dataframe, transform=None):\
self.df, self.transform = dataframe, transform\
def __len__(self): return len(self.df)\
def __getitem__(self, idx):\
img = Image.open(self.df.iloc[idx]['path']).convert('RGB')\
if self.transform: img = self.transform(img)\
return img, torch.tensor(self.df.iloc[idx]['age'], dtype=torch.long)\
\
transform = transforms.Compose([\
transforms.Resize((224, 224)),\
transforms.RandomHorizontalFlip(),\
transforms.ToTensor(),\
transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])\
])\
\
train_loader = DataLoader(AgeDataset(train_df, transform), batch_size=32, shuffle=True)\
val_loader = DataLoader(AgeDataset(val_df, transform), batch_size=32)\
\
best_mae = 99.0\
print(f"\uc0\u55357 \u56960  Lancement de l'entra\'eenement. Les records seront sauv\'e9s dans Projet_IA.")\
\
for epoch in range(20):\
model.train()\
for images, labels in tqdm(train_loader, desc=f"\'c9poque \{epoch+1\}"):\
images, labels = images.to(DEVICE), labels.to(DEVICE)\
optimizer.zero_grad()\
loss = criterion(model(images), labels)\
loss.backward()\
optimizer.step()\
\
model.eval()\
val_mae = 0\
with torch.no_grad():\
for images, labels in val_loader:\
images, labels = images.to(DEVICE), labels.to(DEVICE)\
probs = F.softmax(model(images), dim=1)\
pred_age = (probs * age_range).sum(dim=1)\
val_mae += F.l1_loss(pred_age, labels.float(), reduction='sum').item()\
\
current_mae = val_mae / len(val_df)\
print(f"\uc0\u55356 \u57263  Fin \'c9poque \{epoch+1\} | MAE actuel: \{current_mae:.2f\}")\
\
if current_mae < best_mae:\
best_mae = current_mae\
# SAUVEGARDE DIRECTE SUR LE DRIVE DANS TON DOSSIER Projet_IA\
torch.save(model.state_dict(), SAVE_PATH)\
print(f"\uc0\u55357 \u56510  Record battu ! Sauvegarde effectu\'e9e dans Projet_IA (MAE: \{best_mae:.2f\})")\
}