import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# LLM Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = "gemini-2.0-flash"  # Fast, free, great quality

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MIMIC_DIR = DATA_DIR / "mimic" / "mimic-iv-clinical-database-demo-2.2"
SEPSIS_DIR = DATA_DIR / "mimic" / "sepsis_challenge"
GUIDELINES_DIR = DATA_DIR / "guidelines"
PROCESSED_DIR = DATA_DIR / "processed"
CHROMA_DIR = DATA_DIR / "chroma_db"

# Create directories if they don't exist
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DIR.mkdir(parents=True, exist_ok=True)

# Clinical Thresholds
OUTLIER_Z_THRESHOLD = 3.0  # Z-score for lab outlier detection
TREND_WINDOW_HOURS = 72  # Look-back window for trend analysis
SOFA_SEPSIS_THRESHOLD = 2  # SOFA >= 2 suggests sepsis
QSOFA_ALERT_THRESHOLD = 2  # qSOFA >= 2 is high risk

# Agent Configuration
LLM_TEMPERATURE = 0.1  # Low temperature for medical accuracy
RAG_TOP_K = 5  # Number of guideline chunks to retrieve
CHUNK_SIZE = 500  # Characters per guideline chunk
CHUNK_OVERLAP = 100  # Overlap between chunks
