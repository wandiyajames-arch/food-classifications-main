import pandas as pd
import os

def generate_gt(test_dir, output_path):
    """
    Parcourt le dossier de test structuré en sous-dossiers (ImageFolder)
    et génère le CSV de référence.
    """
    data = []
    
    # On récupère la liste des classes (noms des dossiers) triée par ordre alphabétique
    # C'est ainsi que PyTorch ImageFolder attribue les indices 0, 1, 2...
    classes = sorted([d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))])
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    
    print(f"Mapping détecté : {class_to_idx}")

    for class_name in classes:
        class_path = os.path.join(test_dir, class_name)
        label = class_to_idx[class_name]
        
        for img_name in os.listdir(class_path):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                data.append({
                    "id": img_name,
                    "label": label
                })

    df = pd.DataFrame(data)
    # Tri par ID pour que le leaderboard soit toujours cohérent
    df = df.sort_values('id')
    df.to_csv(output_path, index=False)
    print(f"Fichier de vérité terrain généré : {output_path} ({len(df)} images)")

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # Attention : utilise le dossier de test qui contient les sous-dossiers de classes
    TEST_DATA_DIR = os.path.join(BASE_DIR, "data", "test") 
    GT_OUTPUT_PATH = os.path.join(BASE_DIR, "data", "test_labels.csv")
    
    generate_gt(TEST_DATA_DIR, GT_OUTPUT_PATH)