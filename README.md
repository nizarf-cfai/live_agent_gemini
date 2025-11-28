# Gemini Live API R&D Project

Research & Development project exploring the **Gemini Live API** using **Vertex AI** for real-time audio interactions with medical data.

## Overview

This project demonstrates the capabilities of Google's Gemini Live API (Multimodal Live API) through Vertex AI, featuring:

- **Real-time voice interaction** with native audio input/output
- **Voice Activity Detection (VAD)** for natural conversation flow
- **Proactive audio** with wake word ("Medforce") activation
- **RAG (Retrieval Augmented Generation)** over medical patient data
- **Live transcription** of both user input and agent responses

## Tech Stack

- **Gemini 2.5 Flash** (`gemini-live-2.5-flash-preview-native-audio-09-2025`)
- **Vertex AI API** via `google-genai` SDK
- **PyAudio** for real-time audio streaming
- **Text Embedding 005** for semantic search
- **Python 3.10+**

## Project Structure

```
live_agent/
├── main.py                 # Live agent with audio I/O
├── rag.py                  # RAG system with persistent index
├── canvas_ops.py           # Data fetching from Canvas API
├── system_prompt.md        # Agent persona and instructions
├── requirements.txt        # Python dependencies
├── .env                    # Environment variables (not in repo)
└── output/                 # Cached data and index (not in repo)
    ├── board_items.json    # Medical dashboard data
    └── rag_index.pkl       # Cached embeddings
```

## Features

### 1. Live Audio Agent (`main.py`)
- **Bidirectional streaming** with 16kHz input / 24kHz output
- **Wake word detection** - responds only when "Medforce" is mentioned
- **Real-time transcription** - displays user and agent speech as text
- **Voice Activity Detection** - automatic turn-taking
- **Custom voice** - Uses "Fenrir" voice preset

### 2. RAG System (`rag.py`)
- **Semantic search** over medical patient data
- **Persistent index** with incremental updates
- **Hash-based change detection** - only re-processes modified items
- **Fast loading** - 2-5 seconds on subsequent runs vs 30-60 seconds initial

### 3. Data Integration (`canvas_ops.py`)
- Fetches medical dashboard items from Canvas API
- Local caching with fallback support
- Processes 40+ patient data components

## Setup

### Prerequisites

1. **Google Cloud Project** with Vertex AI API enabled
2. **Application Default Credentials** configured:
   ```bash
   gcloud auth application-default login
   ```

### Installation

1. Clone the repository:
   ```bash
   git clone <repo-url>
   cd live_agent
   ```

2. Create virtual environment:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   source .venv/bin/activate  # Linux/Mac
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Configure environment variables (`.env`):
   ```env
   PROJECT_ID=your-gcp-project-id
   PROJECT_LOCATION=us-central1
   CANVAS_URL=https://your-canvas-api-url
   ```

## Usage

### Run Live Agent

```bash
python main.py
```

**Interaction:**
1. Say "Medforce" to activate the agent
2. Ask medical/clinical questions
3. Agent responds with audio and displays transcription
4. Press `Ctrl+C` to exit

### Build/Update RAG Index

```bash
# First run - builds index
python rag.py

# Subsequent runs - uses cache
python rag.py

# Force rebuild
python rag.py --rebuild
```

### Fetch Latest Data

```bash
python canvas_ops.py
```

## Configuration

### System Prompt (`system_prompt.md`)
Defines the agent's persona and behavior:
- Specializes in medical and clinical information
- Only responds when addressed as "Medforce"
- Maintains professional and empathetic tone

### Audio Settings (`main.py`)
- **Input Rate**: 16000 Hz (required by API)
- **Output Rate**: 24000 Hz (model output)
- **Format**: 16-bit PCM, mono channel
- **Chunk Size**: 512 samples

### VAD Configuration
- **Start Sensitivity**: LOW
- **End Sensitivity**: LOW
- **Prefix Padding**: 200ms
- **Silence Duration**: 300ms

## API Reference

### Gemini Live API
- **Model**: `gemini-live-2.5-flash-preview-native-audio-09-2025`
- **Endpoint**: Vertex AI (`us-central1`)
- **API Version**: `v1beta1`

### Key Features Used
- ✅ Native audio input/output
- ✅ Proactive audio (wake word)
- ✅ Audio transcription (input & output)
- ✅ Voice Activity Detection
- ✅ Custom voice selection
- ✅ System instructions

## Research Findings

### Performance
- **Latency**: ~200-500ms for audio response
- **Transcription Accuracy**: High for medical terminology
- **Wake Word Detection**: Reliable with "Medforce"

### Limitations
- Requires stable internet connection
- Input audio must be 16kHz (API requirement)
- Proactive audio still experimental

### Best Practices
1. Use simple, clear wake words
2. Configure VAD for your environment
3. Cache RAG index for faster startup
4. Monitor API quotas and costs

## Development Status

🔬 **Research & Development** - This is an experimental project exploring Gemini Live API capabilities.

**Current Phase**: Proof of Concept  
**Status**: Functional prototype with core features implemented

## License

Research project - Not for production use

## Acknowledgments

Built with:
- Google Gemini Live API (Vertex AI)
- Google Gen AI SDK for Python
- PyAudio for audio streaming
