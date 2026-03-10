# ⏳ ChronoFace

**Auteurs :** Ouriel Mimoun & David Ceylon  
**Technologies :** Swin Transformer · CycleGAN · Streamlit · PyTorch

ChronoFace est une application web qui permet :
- 🎂 **Prédire l'âge** d'une personne à partir d'une photo (modèle Swin Transformer).
- 👴👶 **Simuler le vieillissement ou le rajeunissement** du visage (CycleGAN entraîné sur UTKFace).

---

## 📋 Prérequis

- Python 3.9+
- Git
- Un environnement virtuel (recommandé)

---

## 🚀 Installation

### 1. Cloner le dépôt

```bash
git clone https://github.com/OuriMims26/Vrai_ChronoFace.git
cd Vrai_ChronoFace
```

### 2. Créer et activer un environnement virtuel

```bash
python -m venv .venv
source .venv/bin/activate      # Mac / Linux
# .venv\Scripts\activate       # Windows
```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

---

## 📥 Télécharger les fichiers de poids (OBLIGATOIRE)

> ⚠️ **L'application ne fonctionnera pas sans ces fichiers.** Ils ne sont pas inclus dans le dépôt en raison de leur taille.

Les 3 fichiers de poids doivent être téléchargés depuis Google Drive et placés **à la racine du projet** (au même niveau que `app.py`) :

| Fichier | Description |
|---|---|
| `checkpoint_fold_3.pth` | Poids du modèle de prédiction d'âge (Swin Transformer) |
| `best_G_AB.pth` | Poids du générateur de vieillissement (CycleGAN : Jeune → Vieux) |
| `G_BA_45.pth` | Poids du générateur de rajeunissement (CycleGAN : Vieux → Jeune) |

📂 **Lien Google Drive :** [Télécharger les 3 fichiers poids](https://drive.google.com/drive/folders/10_EO8AE7Fqk7aJcVDLWeXDiSA9fClN15?usp=drive_link)

Après téléchargement, la structure du projet doit ressembler à ceci :

```
Vrai_ChronoFace/
├── app.py
├── requirements.txt
├── checkpoint_fold_3.pth   ← téléchargé depuis Drive
├── best_G_AB.pth           ← téléchargé depuis Drive
├── G_BA_45.pth             ← téléchargé depuis Drive
└── ...
```

---

## ▶️ Lancer l'application

```bash
streamlit run app.py
```

L'application s'ouvrira automatiquement dans votre navigateur à l'adresse `http://localhost:8501`.

### Utilisation

1. **Uploader** une photo de portrait (formats acceptés : JPG, JPEG, PNG, HEIC).
2. Cliquer sur **"1. Predict Age 🚀"** pour obtenir l'âge estimé.
3. Choisir l'effet souhaité (**👴 Aging** ou **👶 Rejuvenation**) puis cliquer sur **"2. Transform ✨"** pour générer l'image.

---

## 📁 Structure du projet

```
Vrai_ChronoFace/
├── app.py              # Application Streamlit principale
├── gan_model.py        # Architecture du générateur CycleGAN
├── model_training.py   # Script d'entraînement
├── requirements.txt    # Dépendances Python
├── README.md           # Ce fichier
├── README_AGE.md       # Documentation du module de prédiction d'âge
└── README_GAN.md       # Documentation du module CycleGAN
```

---

## 🛠️ Support et compatibilité

- Le modèle d'âge utilise automatiquement **MPS** (Mac Apple Silicon), **CUDA** (GPU NVIDIA) ou **CPU**.
- Le modèle GAN tourne en **CPU** par défaut pour garantir la compatibilité maximale.
