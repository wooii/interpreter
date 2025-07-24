# Interpreter: Meeting Assistant

A real-time transcription and interpretation tool for meetings and conferences using Python and OpenAI's Whisper. This project currently focuses on:
- Transcribing speech in real time, minimizing information loss from audio chunking.
- Translating uncommon words for better understanding.
- Providing key words and summaries for oral speech, assisting second language conversations.

**Note:** Whisper model finetuning functionality will be added in a future update.

## Features
- Real-time audio recording and transcription
- Integration with OpenAI Whisper API or local Whisper models
- Keyword extraction and translation for uncommon words
- Designed for live interpretation and meeting assistance

## Requirements
- Python 3.8+
- [openai-whisper](https://github.com/openai/whisper) or compatible library
- Other dependencies listed in `pyproject.toml`

## Installation
1. Open Terminal and navigate to the directory containing `pyproject.toml`:
   ```sh
   cd /Users/chen/Library/CloudStorage/Dropbox/Code/py/ai/interpreter
   ```
2. Install in development mode:
   ```sh
   pip install -e .
   ```

## Usage
- To run the meeting assistant:
  ```sh
  python interpreter/meeting_assistant.py
  ```
- For testing and development, see scripts in the `interpreter/` directory.

## Roadmap
- [x] Split meeting assistant and finetuning functionalities
- [ ] Implement real-time audio chunking and streaming
- [ ] Integrate Whisper API/local model
- [ ] Add keyword extraction and summary display
- [ ] Build user interface (optional)
- [ ] Add Whisper model finetuning (future)

## License
MIT
