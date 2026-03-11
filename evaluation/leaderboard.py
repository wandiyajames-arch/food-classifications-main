import pandas as pd
import os
from evaluate import evaluate 


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEADERBOARD_FILE = os.path.join(BASE_DIR, "leaderboard", "leaderboard.csv")

def update(team_name, submission_file, true_file):
    # 1. Calcul des scores via module evaluate
    acc, f1 = evaluate(true_file, submission_file)

    # 2. Chargement ou création du leaderboard
    if os.path.exists(LEADERBOARD_FILE):
        leaderboard = pd.read_csv(LEADERBOARD_FILE)
    else:
        # Création du dossier si inexistant
        os.makedirs(os.path.dirname(LEADERBOARD_FILE), exist_ok=True)
        leaderboard = pd.DataFrame(columns=["team", "accuracy", "f1_score"])

    # 3. Gestion de la mise à jour (Best Score Only)
    # On vérifie si l'équipe a déjà un score
    existing_team = leaderboard[leaderboard['team'] == team_name]
    
    if not existing_team.empty:
        old_acc = existing_team['accuracy'].values[0]
        if acc > old_acc:
            print(f"Bravo {team_name} ! Nouveau record personnel : {acc:.4f} > {old_acc:.4f}")
            leaderboard.loc[leaderboard['team'] == team_name, ['accuracy', 'f1_score']] = [acc, f1]
        else:
            print(f"Score reçu pour {team_name} ({acc:.4f}), mais le record précédent était meilleur ({old_acc:.4f}).")
    else:
        # Nouvelle équipe
        new_row = pd.DataFrame({"team": [team_name], "accuracy": [acc], "f1_score": [f1]})
        leaderboard = pd.concat([leaderboard, new_row], ignore_index=True)

    # 4. Tri par accuracy et sauvegarde
    leaderboard = leaderboard.sort_values(by="accuracy", ascending=False)
    leaderboard.to_csv(LEADERBOARD_FILE, index=False)
    
    print("\n--- Current Leaderboard ---")
    print(leaderboard.head(10)) # Affiche le top 10 dans la console

if __name__ == "__main__":
    # On définit BASE_DIR par rapport à ce fichier (evaluation/leaderboard.py)
    # abspath(__file__) donne le chemin vers leaderboard.py
    # dirname donne le dossier evaluation/
    # le second dirname donne la racine du projet food-classifications/
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    BASE_DIR = os.path.dirname(CURRENT_DIR)

    # Construction des chemins absolus
    TEAM_SUB = os.path.join(BASE_DIR, "submissions", "team1.csv")
    TRUE_LABELS = os.path.join(BASE_DIR, "data", "test_labels.csv")

    # Vérification avant de lancer (pour un message d'erreur plus clair)
    if not os.path.exists(TRUE_LABELS):
        print(f"ERREUR : Le fichier vérité est introuvable à : {TRUE_LABELS}")
        print("As-tu bien lancé tools/generate_ground_truth.py avant ?")
    elif not os.path.exists(TEAM_SUB):
        print(f"ERREUR : La soumission est introuvable à : {TEAM_SUB}")
    else:
        update("team1", TEAM_SUB, TRUE_LABELS)