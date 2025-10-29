from datetime import datetime, timedelta
import boto3
import os
import io
import sys
import time
from glonet_forecast import create_forecast
from model import synchronize_model_locally
from get_inits import generate_initial_data
from s3_upload import save_bytes_to_s3

# Parametres par défaut
LOCAL_MODEL_DIR = "/tmp/glonet"

#Fct Principale: qui orchestre toute l'inference
def main():
    print("=" * 60)
    print("GLONET Inference - Processus EDITO")
    print("=" * 60)
    
    # Lire la date depuis une variable d'environnement-La valeur par défaut (hier) est gérée
    default_date_str = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    forecast_date_str_input = os.environ.get("FORECAST_DATE", default_date_str)

    if forecast_date_str_input: # Vérifie si la chaîne n'est ni None ni vide ""
        forecast_date_str = forecast_date_str_input
    else:
        forecast_date_str = default_date_str
        print(f"Date non spécifiée ou vide. Utilisation de la date par défaut (hier) : {forecast_date_str}")
    try:
        FORECAST_DATE = datetime.strptime(forecast_date_str, "%Y-%m-%d").date()
    except ValueError:
        print(f"Erreur : La variable d'environnement FORECAST_DATE ('{forecast_date_str}') n'est pas au format YYYY-MM-DD")
        sys.exit(1)

    print(f"Date de prévision : {FORECAST_DATE}")

    # Lire le dossier S3 depuis une variable d'environnement
    s3_output_folder_input = os.environ.get("S3_OUTPUT_FOLDER")

    if s3_output_folder_input:
        # Si oui, on l'utilise
        S3_OUTPUT_FOLDER = s3_output_folder_input
        print(f"Chemin S3 personnalisé utilisé : {S3_OUTPUT_FOLDER}")
    else:
        # Si non (variable vide ou absente), on construit le chemin par défaut
        print("Aucun chemin S3 personnalisé. Construction du chemin par défaut...")
        
        # On récupère le bucket S3 personnel de l'utilisateur, injecté par EDITO
        user_s3_bucket = os.environ.get("BUCKET_NAME")
        
        if not user_s3_bucket:
            print("Erreur fatale : Impossible de trouver BUCKET_NAME dans l'environnement.")
            print("Le processus ne peut pas déterminer où sauvegarder les résultats.")
            sys.exit(1)
            
        # On construit le chemin par défaut (ex: "oidc-jdupont/glonet-inference")
        S3_OUTPUT_FOLDER = f"{user_s3_bucket}/glonet-inference"
        print(f"Chemin S3 par défaut utilisé : {S3_OUTPUT_FOLDER}")
        
    print(f"\n{'=' * 60}")
    print(f"Demarrage de l'inference...")
    print(f"{'=' * 60}\n")
    
    # Télécharger les modèles
    print("--- Telechargement des modeles ---")
    synchronize_model_locally(LOCAL_MODEL_DIR)
    
    # Calculer les dates: la prevision commence demain et finit dans 10 jrs
    forecast_start = FORECAST_DATE + timedelta(days=1)
    forecast_end = FORECAST_DATE + timedelta(days=10)
    
    # Construire l'URL S3 complète avec la date pour le fichier NetCDF final
    s3_output_with_date = f"{S3_OUTPUT_FOLDER}/{FORECAST_DATE}"
    
    forecast_netcdf_url = (
        f"https://minio.dive.edito.eu/{s3_output_with_date}/"
        f"GLONET_MOI_{forecast_start}_{forecast_end}.nc"
    )
    
    # Générer les données initiales: depuis Copernicus Marine vers S3 (dans bucket)
    print("--- Donnees initiales ---")
    
    bucket_name = S3_OUTPUT_FOLDER.split('/')[0]
    folder_path = S3_OUTPUT_FOLDER.split('/')[1] if len(S3_OUTPUT_FOLDER.split('/'))>1 else ""
    
    in1_url, in2_url, in3_url = generate_initial_data(
        bucket_name=bucket_name,
        forecast_netcdf_file_url=forecast_netcdf_url
    )
    print(f"in1: {in1_url}")
    print(f"in2: {in2_url}")
    print(f"in3: {in3_url}")
    
    # Lancer l'inférence
    print("--- Inference ---")
    dataset = create_forecast(
        forecast_netcdf_file_url=forecast_netcdf_url,
        model_dir=LOCAL_MODEL_DIR,
        initial_file_1_url=in1_url,
        initial_file_2_url=in2_url,
        initial_file_3_url=in3_url,
    )
    
    try:
        #Nom du fichier final
        file_name = f"GLONET_MOI_{forecast_start}_{forecast_end}.nc"
        file_key = f"{folder_path}/{FORECAST_DATE}/{file_name}"        #Construire le chemin complet dans S3
        file_key = file_key.lstrip('/')        
        print("--- Sauvegarde des resultats sur S3 (Streaming) ---")
        print(f"Conversion du dataset en NetCDF (en mémoire)...")

        # Convertir le dataset en format NetCDF DANS LA RAM
        object_bytes = dataset.to_netcdf()
        
        #Verifier le type et avoir des bytes a la fin
        if isinstance(object_bytes, memoryview):
            object_bytes = bytes(object_bytes)
            
        print(f"Taille en mémoire : {len(object_bytes) / 1e6:.2f} MB")

        # Créer un "fichier virtuel" en RAM à partir de ces bytes 
        in_memory_file = io.BytesIO(object_bytes)

        # Créer le client S3 (utilise tes variables d'environnement)
        s3_endpoint_url = os.environ.get("S3_ENDPOINT")
        if s3_endpoint_url and not s3_endpoint_url.startswith("https://"):
            s3_endpoint_url = "https://" + s3_endpoint_url
        
        s3_client = boto3.client(
            "s3",
            endpoint_url=s3_endpoint_url,
            aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
            aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
            aws_session_token=os.environ["AWS_SESSION_TOKEN"],
        )
        
        print(f"Sauvegarde en streaming (multipart) vers S3...")
        print(f"  Bucket: {bucket_name}")
        print(f"  Chemin: {file_key}")

        # Utiliser upload_fileobj : gère les gros fichiers depuis la RAM: decoupe le fichier en chunks et les envoie successiveemnt
        s3_client.upload_fileobj(
            in_memory_file,     # L'objet "fichier" en mémoire
            bucket_name ,       # Le bucket
            file_key            # Le chemin S3
        )

        result_url = f"s3://{bucket_name}/{file_key}"
        print(f"Resultats sauvegardes : {result_url}\n")

    except Exception as e:
        print(f"Erreur lors de la sauvegarde : {e}")
        raise
    
    # Résumé final
    print("=" * 60)
    print("Inference terminee avec succes !")
    print(f"  Date : {FORECAST_DATE}")
    print(f"  Resultats : {result_url}")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nErreur : {e}")
        sys.exit(1)