import assemblyai as aai
import spacy
import os
from dotenv import load_dotenv
from collections import defaultdict

# Load environment variables from .env file
load_dotenv()

# Get API key from environment variable
api_key = os.getenv("ASSEMBLYAI_API_KEY")
if api_key:
    aai.settings.api_key = api_key
    print("API key loaded from environment variable")
else:
    # Fallback for development or error handling
    raise ValueError("ASSEMBLYAI_API_KEY not found in environment variables")

# Initialize spaCy
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    import subprocess
    import sys
    subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

def analyze_sentiment_spacy(text):
    """Analyze sentiment using spaCy"""
    if not text or len(text.strip()) < 3:
        return "NEUTRAL"
    
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

def get_audio_insights(audio_file):
    """
    This function takes an audio file and returns the transcript,
    speaker labels, and sentiment analysis results.
    """
    try:
        # Create a temporary file
        with open("temp_audio.mp3", "wb") as f:
            f.write(audio_file.getbuffer())
        
        # Configuration for speaker diarization
        config = aai.TranscriptionConfig(
            speaker_labels=True,
            sentiment_analysis=False
        )

        transcriber = aai.Transcriber()
        transcript = transcriber.transcribe("temp_audio.mp3", config)

        # Clean up temporary file
        if os.path.exists("temp_audio.mp3"):
            os.remove("temp_audio.mp3")

        if transcript.status == aai.TranscriptStatus.error:
            return "", "", [], f"Transcription error: {transcript.error}"
        
        if not transcript.utterances:
            return transcript.text, "", [], "No utterances found"

        # Extracting speaker information
        speaker_text = ""
        sentiments = []
        
        for utterance in transcript.utterances:
            speaker_text += f"Speaker {utterance.speaker}: {utterance.text}\n"

            # Analyze sentiment using spaCy
            sentiment = analyze_sentiment_spacy(utterance.text)
            
            sentiments.append({
                "text": utterance.text,
                "speaker": utterance.speaker,
                "sentiment": sentiment,
            })

        return transcript.text, speaker_text, sentiments, None

    except Exception as e:
        # Clean up temporary file if it existss
        if os.path.exists("temp_audio.mp3"):
            os.remove("temp_audio.mp3")
        return "", "", [], f"An error occurred: {str(e)}"