import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

def evaluate(true_csv_path, pred_csv_path):
    """
    Compare le fichier de vérité terrain avec la soumission du participant.
    """
    # 1. Chargement des fichiers
    df_true = pd.read_csv(true_csv_path)
    df_pred = pd.read_csv(pred_csv_path)

    # 2. Vérification du format
    if 'id' not in df_pred.columns or 'prediction' not in df_pred.columns:
        raise ValueError("Le fichier de soumission doit contenir les colonnes 'id' et 'prediction'.")

    # 3. Alignement des données
    # On s'assure que les lignes sont dans le même ordre selon l'ID
    df_true = df_true.sort_values('id').reset_index(drop=True)
    df_pred = df_pred.sort_values('id').reset_index(drop=True)

    # 4. Vérification de l'intégrité
    if len(df_true) != len(df_pred):
        raise ValueError(f"Nombre de lignes incorrect : attendu {len(df_true)}, reçu {len(df_pred)}.")
    
    if not df_true['id'].equals(df_pred['id']):
        # À ajouter pour debugger dans evaluate.py
        ids_true = set(df_true['id'])
        ids_pred = set(df_pred['id'])

        print(f"IDs en trop dans la soumission : {ids_pred - ids_true}")
        print(f"IDs manquants dans la soumission : {ids_true - ids_pred}")
        raise ValueError("Les IDs des images dans le fichier de soumission ne correspondent pas à la vérité terrain.")

    # 5. Calcul des scores
    y_true = df_true['label'] # Supposé être la colonne 'label' dans ton test_labels.csv
    y_pred = df_pred['prediction']

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")


    return acc, f1


