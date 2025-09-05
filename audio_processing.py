import assemblyai as aai
import spacy
import os
from dotenv import load_dotenv
import subprocess
import sys

# Load environment variables from .env file
load_dotenv()

# Get API key from environment variable
api_key = os.getenv("ASSEMBLYAI_API_KEY")
if api_key:
    aai.settings.api_key = api_key
    print("API key loaded from environment variable")
else:
    raise ValueError("ASSEMBLYAI_API_KEY not found in environment variables")

# Initialize spaCy with better error handling
def load_spacy_model():
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        print("Downloading spaCy model...")
        # Try to download the model
        try:
            subprocess.run([
                sys.executable, 
                "-m", "spacy", 
                "download", "en_core_web_sm"
            ], check=True, capture_output=True)
            return spacy.load("en_core_web_sm")
        except (subprocess.CalledProcessError, OSError):
            print("Failed to download spaCy model. Using basic sentiment analysis.")
            return None

nlp = load_spacy_model()

def analyze_sentiment_spacy(text):
    """Analyze sentiment using spaCy or fallback to basic analysis"""
    if not text or len(text.strip()) < 3:
        return "NEUTRAL"
    
    # If spaCy model failed to load, use basic analysis
    if nlp is None:
        return basic_sentiment_analysis(text)
    
    try:
        doc = nlp(text)
        
        positive_words = {'good', 'great', 'excellent', 'awesome', 'wonderful', 
                         'amazing', 'fantastic', 'perfect', 'love', 'like', 'happy'}
        negative_words = {'bad', 'terrible', 'awful', 'horrible', 'hate', 
                         'dislike', 'sad', 'angry', 'frustrated', 'disappointed'}
        
        pos_count = 0
        neg_count = 0
        
        for token in doc:
            if token.text.lower() in positive_words:
                pos_count += 1
            elif token.text.lower() in negative_words:
                neg_count += 1
        
        if pos_count > neg_count:
            return "POSITIVE"
        elif neg_count > pos_count:
            return "NEGATIVE"
        elif pos_count == neg_count and pos_count > 0:
            return "MIXED"
        else:
            return "NEUTRAL"
    except Exception:
        return basic_sentiment_analysis(text)

def basic_sentiment_analysis(text):
    """Fallback sentiment analysis without spaCy"""
    if not text:
        return "NEUTRAL"
    
    text_lower = text.lower()
    positive_words = {'good', 'great', 'excellent', 'awesome', 'wonderful', 
                     'amazing', 'fantastic', 'perfect', 'love', 'like', 'happy'}
    negative_words = {'bad', 'terrible', 'awful', 'horrible', 'hate', 
                     'dislike', 'sad', 'angry', 'frustrated', 'disappointed'}
    
    pos_count = sum(1 for word in positive_words if word in text_lower)
    neg_count = sum(1 for word in negative_words if word in text_lower)
    
    if pos_count > neg_count:
        return "POSITIVE"
    elif neg_count > pos_count:
        return "NEGATIVE"
    elif pos_count == neg_count and pos_count > 0:
        return "MIXED"
    else:
        return "NEUTRAL"

# ... rest of your get_audio_insights function ...