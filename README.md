# Audio AI Information Extractor

A Streamlit app that transcribes audio files, identifies speakers, and analyzes sentiment using AssemblyAI and spaCy.

## Setup

1. Clone this repository
2. Install dependencies: `pip install -r requirements.txt`
3. Copy `.env.example` to `.env`: `cp .env.example .env`
4. Add your AssemblyAI API key to the `.env` file
5. Run: `streamlit run app.py`

## Getting an AssemblyAI API Key

1. Sign up at [AssemblyAI](https://www.assemblyai.com/)
2. Go to your dashboard and copy your API key
3. Add it to your `.env` file: `ASSEMBLYAI_API_KEY=your_actual_key_here`

## Deployment

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://yourapp-name.streamlit.app/)
