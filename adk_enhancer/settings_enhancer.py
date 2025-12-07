import os
import vertexai
from vertexai._genai.client import Client


# =====================
# Variables
# =====================
GOOGLE_CLOUD_PROJECT: str = os.getenv("GOOGLE_CLOUD_PROJECT", None)
GOOGLE_CLOUD_LOCATION: str = os.getenv("GOOGLE_CLOUD_LOCATION", None)
GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", None)


VERTEXAI_CLIENT: Client = vertexai.Client(
    # api_key=GOOGLE_API_KEY,
    project=GOOGLE_CLOUD_PROJECT, 
    location=GOOGLE_CLOUD_LOCATION
)
