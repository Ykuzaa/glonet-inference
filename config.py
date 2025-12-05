"""
Centralized configuration for the GLONET inference application.

This module reads configuration parameters from environment variables
and provides them as constants for other modules to use. This approach
centralizes configuration management, making the application easier
to maintain and configure.
"""
import os
import sys
from datetime import datetime, timedelta

# --- S3/AWS Configuration ---
AWS_S3_ENDPOINT = os.environ.get("AWS_S3_ENDPOINT")
AWS_BUCKET_NAME = os.environ.get("AWS_BUCKET_NAME")
AWS_ACCESS_KEY_ID = os.environ.get("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY")
AWS_SESSION_TOKEN = os.environ.get("AWS_SESSION_TOKEN")

# --- Copernicus Marine Configuration ---
COPERNICUSMARINE_SERVICE_USERNAME = os.environ.get("COPERNICUSMARINE_SERVICE_USERNAME")
COPERNICUSMARINE_SERVICE_PASSWORD = os.environ.get("COPERNICUSMARINE_SERVICE_PASSWORD")

# --- Execution Configuration ---

# Forecast date (defaults to yesterday)
DEFAULT_DATE_STR = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
FORECAST_DATE_STR = os.environ.get("FORECAST_DATE", DEFAULT_DATE_STR)
try:
    FORECAST_DATE = datetime.strptime(FORECAST_DATE_STR, "%Y-%m-%d").date()
except ValueError:
    print(f"Error: Environment variable FORECAST_DATE ('{FORECAST_DATE_STR}') is not in YYYY-MM-DD format", file=sys.stderr)
    sys.exit(1)

# S3 output folder (defaults to a path within AWS_BUCKET_NAME)
S3_OUTPUT_FOLDER = os.environ.get("S3_OUTPUT_FOLDER")
if not S3_OUTPUT_FOLDER:
    if AWS_BUCKET_NAME:
        S3_OUTPUT_FOLDER = f"{AWS_BUCKET_NAME}/glonet-inference"
    else:
        print("Fatal error: S3_OUTPUT_FOLDER or AWS_BUCKET_NAME must be set.", file=sys.stderr)
        sys.exit(1)