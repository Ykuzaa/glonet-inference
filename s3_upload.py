"""
S3 upload utilities for GLONET inference.

This module provides helpers to construct an S3 endpoint URL,
create a boto3 S3 client, and perform common S3 operations such as listing,
deleting and uploading objects.
"""

import boto3
import os


# Return the S3 endpoint URL, normalizing input and ensuring HTTPS is used.
def get_s3_endpoint_url_with_protocol():
    s3_endpoint_url = os.environ.get("S3_ENDPOINT")

    if not s3_endpoint_url:
        raise ValueError("The environment variable S3_ENDPOINT is not set.")

    s3_endpoint_url = s3_endpoint_url.replace("https://", "").replace(
        "http://", ""
    ).rstrip("/")

    s3_endpoint_url_final = f"https://{s3_endpoint_url}"

    return s3_endpoint_url_final


# Create and return a boto3 S3 client configured with the endpoint and env credentials.
def get_s3_client():
    s3_endpoint_url_final = get_s3_endpoint_url_with_protocol()

    return boto3.client(
        "s3",
        endpoint_url=s3_endpoint_url_final,
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        aws_session_token=os.environ["AWS_SESSION_TOKEN"],
    )


# List objects in a bucket under the given prefix and return the response contents.
def list_objects(bucket_name: str, prefix_key: str) -> list[dict[str, str]]:
    try:
        response = get_s3_client().list_objects(
            Bucket=bucket_name,
            Prefix=prefix_key,
        )
        if response["ResponseMetadata"]["HTTPStatusCode"] == 200:
            print(f"Successfully listed objects: {bucket_name}/{prefix_key}")
            return response["Contents"]
        else:
            raise Exception(response)
    except Exception as exception:
        print(f"Failed to list objects {bucket_name}/{prefix_key}: {exception}")
        raise


# Delete a specific object from the given bucket.
def delete_object(bucket_name: str, object_key: str):
    try:
        response = get_s3_client().delete_object(
            Bucket=bucket_name,
            Key=object_key,
        )
        if response["ResponseMetadata"]["HTTPStatusCode"] in [200, 204]:
            print(f"Successfully deleted object: {bucket_name}/{object_key}")
        else:
            print(f"Failed to delete object {bucket_name}/{object_key}: {response}")
    except Exception as exception:
        print(f"Failed to delete object {bucket_name}/{object_key}: {exception}")
        raise


# Upload raw bytes to S3 as an object at the specified key.
def save_bytes_to_s3(bucket_name: str, object_bytes, object_key: str):
    try:
        response = get_s3_client().put_object(
            Bucket=bucket_name,
            Body=object_bytes,
            Key=object_key,
        )
        if response["ResponseMetadata"]["HTTPStatusCode"] == 200:
            print(f"Successfully uploaded bytes to S3: {bucket_name}/{object_key}")
        else:
            raise Exception(response)
    except Exception as exception:
        print(
            f"Failed to upload bytes to S3 {bucket_name}/{object_key}: {exception}"
        )
        raise


# Upload a local file from disk to S3 at the specified key.
def save_file_to_s3(bucket_name: str, local_file_path: str, object_key: str):
    """Upload a local file to S3."""
    try:
        s3_client = get_s3_client()
        print(f"Uploading {local_file_path} to s3://{bucket_name}/{object_key}...")
        s3_client.upload_file(
            local_file_path,  # source file path on disk
            bucket_name,      # destination bucket
            object_key,       # destination S3 key
        )
        print(f"Upload succeeded to s3://{bucket_name}/{object_key}")
    except Exception as exception:
        print(
            f"Failed to upload file {local_file_path} to s3://{bucket_name}/{object_key}: {exception}"
        )
        raise
