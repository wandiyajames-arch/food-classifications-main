import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np
import os
import sys

# ---------------- Configuration ----------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'
num_classes = 10
model_type = 'simple_cnn'  # ou 'resnet18_transfer'

# Gestion des chemins
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "baseline_model.pth")
TEST_DIR = os.path.join(BASE_DIR, "data", "test")
TRUE_LABELS_CSV = os.path.join(BASE_DIR, "data", "test_labels.csv")
SUBMISSION_PATH = os.path.join(BASE_DIR, "submissions", "team1.csv")

# Importation dynamique du modèle (on s'assure que le dossier src est dans le path)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model_baseline import simple_cnn, resnet18_transfer

# ---------------- Dataset Robuste ----------------
class RobustTestDataset(Dataset):
    """Charge uniquement les images listées dans le fichier de vérité terrain."""
    def __init__(self, csv_file, root_dir, transform=None):
        self.df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.iloc[idx, 0] # Récupère la colonne 'id'
        
        # Recherche récursive de l'image dans TEST_DIR
        img_path = None
        for root, dirs, files in os.walk(self.root_dir):
            if img_name in files:
                img_path = os.path.join(root, img_name)
                break
        
        if img_path is None:
            raise FileNotFoundError(f"Impossible de trouver l'image : {img_name}")

        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, img_name

# ---------------- Fonctions utiles ----------------
def load_model(model_path, num_classes, model_type, device):
    if model_type == 'simple_cnn':
        model = simple_cnn(num_classes)
    elif model_type == 'resnet18_transfer':
        model = resnet18_transfer(num_classes)
    else:
        raise ValueError(f"Modèle inconnu : {model_type}")
    
    # map_location permet de charger un modèle GPU sur un CPU sans erreur
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()
    return model

# ---------------- Main Script ----------------
if __name__ == "__main__":
    # 1. Transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    # 2. Load Model
    if not os.path.exists(MODEL_PATH):
        print(f"Erreur : Modèle introuvable à {MODEL_PATH}")
        sys.exit()

    model = load_model(MODEL_PATH, num_classes, model_type, device)
    print(f"Modèle chargé avec succès.")

    # 3. Load Dataset
    if not os.path.exists(TRUE_LABELS_CSV):
        print(f"Erreur : Générez d'abord le fichier {TRUE_LABELS_CSV} avec tools/generate_gt.py")
        sys.exit()

    test_dataset = RobustTestDataset(TRUE_LABELS_CSV, TEST_DIR, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # 4. Inférence
    all_predictions = []
    all_ids = []

    print(f"Début de l'inférence sur {len(test_dataset)} images...")
    with torch.no_grad():
        for images, names in test_loader:
            images = images.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            all_predictions.extend(preds.cpu().numpy())
            all_ids.extend(names)

    # 5. Sauvegarde
    os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)
    submission_df = pd.DataFrame({
        'id': all_ids,
        'prediction': all_predictions
    })
    
    submission_df.to_csv(SUBMISSION_PATH, index=False)
    print(f"Soumission sauvegardée : {SUBMISSION_PATH}")
    print(f"Nombre de lignes : {len(submission_df)}")