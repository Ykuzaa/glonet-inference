# --- Fichier: model.py ---

import os
from pathlib import Path
import boto3
# Pas besoin de botocore ou Config pour l'accès authentifié

# --- Configuration ---
# Dossier dans le bucket personnel contenant les modèles
MODELS_PREFIX_IN_YOUR_BUCKET = "glonet-models/"
# --------------------

def synchronize_model_locally(local_dir: str):
    """Point d'entrée: synchronise les modèles du bucket perso vers local_dir."""
    # Récupère le nom du bucket depuis l'environnement
    user_s3_bucket = os.environ.get("BUCKET_NAME")
    if not user_s3_bucket:
        print("Erreur : BUCKET_NAME manquant.")
        raise ValueError("BUCKET_NAME manquant dans l'environnement")

    print(f"Synchro depuis : s3://{user_s3_bucket}/{MODELS_PREFIX_IN_YOUR_BUCKET}")
    # Appelle la fonction de synchronisation effective
    sync_s3_to_local(
        user_s3_bucket, MODELS_PREFIX_IN_YOUR_BUCKET, local_dir
    )

def sync_s3_to_local(bucket_name, remote_prefix, local_dir):
    """Télécharge les fichiers depuis S3 (authentifié) si non présents localement."""
    # Assure que le dossier local existe
    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    # Crée un client S3 authentifié
    try:
        # Importe la fonction helper pour obtenir le client
        from s3_upload import get_s3_client
        s3_client = get_s3_client()
        print("Client S3 authentifié créé.")
    except Exception as e:
        print(f"Erreur création client S3 : {e}")
        raise

    print(f"Vérification/Téléchargement : s3://{bucket_name}/{remote_prefix} -> {local_dir}...")

    # Liste des fichiers modèles requis
    files_to_download = [
        "L0/zos_mean.npy", "L0/zos_std.npy", "L0/thetao_mean.npy", "L0/thetao_std.npy", "L0/so_mean.npy", "L0/so_std.npy", "L0/uo_mean.npy", "L0/uo_std.npy", "L0/vo_mean.npy", "L0/vo_std.npy",
        "L50/thetao_mean.npy", "L50/thetao_std.npy", "L50/so_mean.npy", "L50/so_std.npy", "L50/uo_mean.npy", "L50/uo_std.npy", "L50/vo_mean.npy", "L50/vo_std.npy",
        "L100/thetao_mean.npy", "L100/thetao_std.npy", "L100/so_mean.npy", "L100/so_std.npy", "L100/uo_mean.npy", "L100/uo_std.npy", "L100/vo_mean.npy", "L100/vo_std.npy",
        "L150/thetao_mean.npy", "L150/thetao_std.npy", "L150/so_mean.npy", "L150/so_std.npy", "L150/uo_mean.npy", "L150/uo_std.npy", "L150/vo_mean.npy", "L150/vo_std.npy",
        "L222/thetao_mean.npy", "L222/thetao_std.npy", "L222/so_mean.npy", "L222/so_std.npy", "L222/uo_mean.npy", "L222/uo_std.npy", "L222/vo_mean.npy", "L222/vo_std.npy",
        "L318/thetao_mean.npy", "L318/thetao_std.npy", "L318/so_mean.npy", "L318/so_std.npy", "L318/uo_mean.npy", "L318/uo_std.npy", "L318/vo_mean.npy", "L318/vo_std.npy",
        "L380/thetao_mean.npy", "L380/thetao_std.npy", "L380/so_mean.npy", "L380/so_std.npy", "L380/uo_mean.npy", "L380/uo_std.npy", "L380/vo_mean.npy", "L380/vo_std.npy",
        "L450/thetao_mean.npy", "L450/thetao_std.npy", "L450/so_mean.npy", "L450/so_std.npy", "L450/uo_mean.npy", "L450/uo_std.npy", "L450/vo_mean.npy", "L450/vo_std.npy",
        "L540/thetao_mean.npy", "L540/thetao_std.npy", "L540/so_mean.npy", "L540/so_std.npy", "L540/uo_mean.npy", "L540/uo_std.npy", "L540/vo_mean.npy", "L540/vo_std.npy",
        "L640/thetao_mean.npy", "L640/thetao_std.npy", "L640/so_mean.npy", "L640/so_std.npy", "L640/uo_mean.npy", "L640/uo_std.npy", "L640/vo_mean.npy", "L640/vo_std.npy",
        "L763/thetao_mean.npy", "L763/thetao_std.npy", "L763/so_mean.npy", "L763/so_std.npy", "L763/uo_mean.npy", "L763/uo_std.npy", "L763/vo_mean.npy", "L763/vo_std.npy",
        "L902/thetao_mean.npy", "L902/thetao_std.npy", "L902/so_mean.npy", "L902/so_std.npy", "L902/uo_mean.npy", "L902/uo_std.npy", "L902/vo_mean.npy", "L902/vo_std.npy",
        "L1245/thetao_mean.npy", "L1245/thetao_std.npy", "L1245/so_mean.npy", "L1245/so_std.npy", "L1245/uo_mean.npy", "L1245/uo_std.npy", "L1245/vo_mean.npy", "L1245/vo_std.npy",
        "L1684/thetao_mean.npy", "L1684/thetao_std.npy", "L1684/so_mean.npy", "L1684/so_std.npy", "L1684/uo_mean.npy", "L1684/uo_std.npy", "L1684/vo_mean.npy", "L1684/vo_std.npy",
        "L2225/thetao_mean.npy", "L2225/thetao_std.npy", "L2225/so_mean.npy", "L2225/so_std.npy", "L2225/uo_mean.npy", "L2225/uo_std.npy", "L2225/vo_mean.npy", "L2225/vo_std.npy",
        "L3220/thetao_mean.npy", "L3220/thetao_std.npy", "L3220/so_mean.npy", "L3220/so_std.npy", "L3220/uo_mean.npy", "L3220/uo_std.npy", "L3220/vo_mean.npy", "L3220/vo_std.npy",
        "L3597/thetao_mean.npy", "L3597/thetao_std.npy", "L3597/so_mean.npy", "L3597/so_std.npy", "L3597/uo_mean.npy", "L3597/uo_std.npy", "L3597/vo_mean.npy", "L3597/vo_std.npy",
        "L3992/thetao_mean.npy", "L3992/thetao_std.npy", "L3992/so_mean.npy", "L3992/so_std.npy", "L3992/uo_mean.npy", "L3992/uo_std.npy", "L3992/vo_mean.npy", "L3992/vo_std.npy",
        "L4405/thetao_mean.npy", "L4405/thetao_std.npy", "L4405/so_mean.npy", "L4405/so_std.npy", "L4405/uo_mean.npy", "L4405/uo_std.npy", "L4405/vo_mean.npy", "L4405/vo_std.npy",
        "L4833/thetao_mean.npy", "L4833/thetao_std.npy", "L4833/so_mean.npy", "L4833/so_std.npy", "L4833/uo_mean.npy", "L4833/uo_std.npy", "L4833/vo_mean.npy", "L4833/vo_std.npy",
        "L5274/thetao_mean.npy", "L5274/thetao_std.npy", "L5274/so_mean.npy", "L5274/so_std.npy", "L5274/uo_mean.npy", "L5274/uo_std.npy", "L5274/vo_mean.npy", "L5274/vo_std.npy",
        "glonet_p1.pt", "glonet_p2.pt", "glonet_p3.pt",
        "ref1.nc", "ref2.nc", "ref3.nc",
        "xe_weights14/L0.nc", "xe_weights14/L50.nc", "xe_weights14/L100.nc", "xe_weights14/L150.nc",
        "xe_weights14/L222.nc", "xe_weights14/L318.nc", "xe_weights14/L380.nc", "xe_weights14/L450.nc",
        "xe_weights14/L540.nc", "xe_weights14/L640.nc", "xe_weights14/L763.nc", "xe_weights14/L902.nc",
        "xe_weights14/L1245.nc", "xe_weights14/L1684.nc", "xe_weights14/L2225.nc", "xe_weights14/L3220.nc",
        "xe_weights14/L3597.nc", "xe_weights14/L3992.nc", "xe_weights14/L4405.nc", "xe_weights14/L4833.nc", "xe_weights14/L5274.nc",
    ]

    # Boucle sur chaque fichier à télécharger
    for file_name in files_to_download:
        # Construit le chemin local complet
        local_file_path = local_dir / file_name
        # Crée le dossier parent local si nécessaire
        local_file_path.parent.mkdir(parents=True, exist_ok=True)

        # Construit la clé S3 (chemin dans le bucket)
        object_key = f"{remote_prefix}{file_name}"

        # Ne télécharge que si le fichier n'existe pas localement
        if not local_file_path.exists():
            try:
                print(f"Téléchargement: s3://{bucket_name}/{object_key} -> {local_file_path}...")
                # Appel Boto3 pour télécharger
                s3_client.download_file(
                    bucket_name,
                    object_key,
                    str(local_file_path)
                )
                print(f"Téléchargé: {file_name}")
            # Gestion des erreurs de téléchargement
            except Exception as e:
                print(f"ERREUR : Téléchargement échoué pour {file_name} depuis s3://{bucket_name}/{object_key}")
                print(f"Vérifier si le fichier existe et les permissions.")
                print(f"Erreur Boto3 : {e}")
                # Arrête le script si un modèle manque
                raise SystemExit(f"Modèle manquant : {file_name}")

    # Message de fin
    print(f"Synchronisation terminée dans {local_dir}")