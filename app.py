import streamlit as st
from audio_processing import get_audio_insights
import tempfile
import os

st.set_page_config(layout="wide")

st.title("AI Powered Audio Information Extractor")
st.write("Upload an audio file to transcribe it, identify speakers, and analyze its sentiment.")

uploaded_file = st.file_uploader("Choose an audio file...", type=["wav", "mp3", "m4a"])

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')

    with st.spinner('Analyzing audio... This might take a moment.'):
        # Call the function and handle the response safely
        result = get_audio_insights(uploaded_file)
        
        # Check if result is valid before unpacking
        if result is None or len(result) != 4:
            st.error("Unexpected error: Function returned invalid data")
            st.stop()
            
        full_transcript, speaker_transcript, sentiments, error = result

    if error:
        st.error(f"An error occurred: {error}")
    else:
        st.header("Analysis Results")

        # Display the extracted information in tabs
        tab1, tab2, tab3 = st.tabs(["Full Transcript", "Speaker Diarization", "Sentiment Analysis"])

        with tab1:
            st.subheader("Full Transcription")
            st.write(full_transcript)

        with tab2:
            st.subheader("Conversation by Speaker")
            st.text(speaker_transcript)

        with tab3:
            st.subheader("Sentiment Analysis")
            if sentiments:
                for sentiment in sentiments:
                    st.write(f"**Speaker {sentiment['speaker']}**: {sentiment['text']}")
                    sentiment_value = sentiment['sentiment']

                    if sentiment_value == "POSITIVE":
                        st.success(f"Sentiment: {sentiment_value}")
                    elif sentiment_value == "NEGATIVE":
                        st.error(f"Sentiment: {sentiment_value}")
                    else:
                        st.info(f"Sentiment: {sentiment_value}")
                    st.write("---")
            else:
                st.info("No sentiment analysis results available.")