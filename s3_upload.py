import boto3
import os


def get_s3_endpoint_url_with_protocol():
    s3_endpoint_url = os.environ.get("S3_ENDPOINT")
    
    # Vérifie si la variable est définie
    if not s3_endpoint_url:
        raise ValueError("La variable d'environnement S3_ENDPOINT n'est pas définie.")

    # Nettoie d'abord les préfixes HTTP/HTTPS (juste pour être sûr)
    s3_endpoint_url = s3_endpoint_url.replace("https://", "").replace("http://", "").rstrip('/')
    
    # AJOUTE TOUJOURS LE PROTOCOLE HTTPS
    s3_endpoint_url_final = f"https://{s3_endpoint_url}"
    
    return s3_endpoint_url_final

def get_s3_client():
    # Récupère l'URL corrigée avec https://
    s3_endpoint_url_final = get_s3_endpoint_url_with_protocol()
    
    return boto3.client(
        "s3",
        endpoint_url=s3_endpoint_url_final,  # Utiliser la variable corrigée
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        aws_session_token=os.environ["AWS_SESSION_TOKEN"],
    )

def list_objects(bucket_name: str, prefix_key: str) -> list[dict[str, str]]:
    try:
        response = get_s3_client().list_objects(
            Bucket=bucket_name,
            Prefix=prefix_key,
        )
        if response["ResponseMetadata"]["HTTPStatusCode"] == 200:
            print(f"Successfully list objects {bucket_name}/{prefix_key}")
            return response["Contents"]
        else:
            raise Exception(response)
    except Exception as exception:
        print(f"Failed to list object {bucket_name}/{prefix_key}: {exception}")
        raise


def delete_object(bucket_name: str, object_key: str):
    try:
        response = get_s3_client().delete_object(
            Bucket=bucket_name,
            Key=object_key,
        )
        if response["ResponseMetadata"]["HTTPStatusCode"] in [200, 204]:
            print(f"Successfully deleted object {bucket_name}/{object_key}")
        else:
            print(
                f"Failed to delete object {bucket_name}/{object_key}: {response}"
            )
    except Exception as exception:
        print(
            f"Failed to delete object {bucket_name}/{object_key}: {exception}"
        )
        raise


def save_bytes_to_s3(bucket_name: str, object_bytes, object_key: str):
    try:
        response = get_s3_client().put_object(
            Bucket=bucket_name,
            Body=object_bytes,
            Key=object_key,
        )
        if response["ResponseMetadata"]["HTTPStatusCode"] == 200:
            print(
                f"Successfully uploaded bytes to S3 {bucket_name}/{object_key}"
            )
        else:
            raise Exception(response)
    except Exception as exception:
        print(
            f"Failed to upload bytes to S3 {bucket_name}/{object_key}: {exception}"
        )
        raise
def save_file_to_s3(bucket_name: str, local_file_path: str, object_key: str):
    """Upload un fichier depuis le disque local vers S3."""
    try:
        s3_client = get_s3_client()
        print(f"Uploading {local_file_path} vers s3://{bucket_name}/{object_key}...")
        s3_client.upload_file(
            local_file_path, # Chemin du fichier source sur disque
            bucket_name,     # Bucket destination
            object_key       # Clé S3 destination
        )
        print(f"Upload réussi vers s3://{bucket_name}/{object_key}")
    except Exception as exception:
        print(f"Échec de l'upload du fichier {local_file_path} vers s3://{bucket_name}/{object_key}: {exception}")
        raise