from s3_upload import list_objects
import os

# Assure-toi que tes variables d'env sont chargées
# (Tu devras peut-être les sourcer avant de lancer le script)

BUCKET = "oidc-emboulaalam"
FOLDER = "glonet-inference/"

print(f"--- Vérification du Bucket: {BUCKET}, Dossier: {FOLDER} ---")

try:
    # Appelle la fonction de ton propre fichier
    objects = list_objects(BUCKET, FOLDER)

    if not objects:
        print("Le dossier est vide ou n'existe pas.")
    else:
        print(f"Trouvé {len(objects)} fichiers :")
        for obj in objects:
            # 'Key' est le chemin complet, 'Size' est la taille
            print(f"  - {obj['Key']}  (Taille: {obj['Size'] / 1e6:.2f} MB)")

except Exception as e:
    print(f"\nErreur lors de la connexion à S3 : {e}")
    print("Vérifie que tes variables d'environnement S3 sont bien chargées !")