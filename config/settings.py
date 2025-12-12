import os

BASE_URL = "https://agents-workshop-backend.cfapps.eu10-004.hana.ondemand.com"

USERNAME = os.getenv("API_USERNAME")
PASSWORD = os.getenv("API_PASSWORD")

OLLAMA_MODEL = "llama3.2"
