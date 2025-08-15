# Interpreter: Real-Time Meeting Assistant

A real-time audio recording, transcription, and interpretation tool for meetings and live conversations. This project is designed for scenarios where speed and accuracy are critical, such as live interpretation or meeting environments.

## Key Features
- Real-time audio recording and transcription with continuous listening
- Speaker identification: print out the words received and indicate who is speaking
- Option to use either advanced speech recognition APIs (OpenAI Whisper or Google Speech Recognition) or local models for transcription
- Highlight keywords and uncommon words in the transcript
- Translate highlighted/uncommon words for better understanding
- Designed for live interpretation, meeting assistance, and second language conversations

## Requirements
- Python 3.8+
- [openai-whisper](https://github.com/openai/whisper)
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
- To run the real-time interpreter:
  ```sh
  python interpreter/main.py
  ```
- To compare different speech recognition models:
  ```sh
  python interpreter/model_comparison.py
  ```

## Roadmap
- [x] Implement real-time, continuous audio recording and streaming
- [x] Integrate Whisper API/local model, and evaluate other advanced models
- [ ] Add speaker identification and display who is talking
- [ ] Highlight and translate keywords/uncommon words in real time
- [ ] Build user interface (optional)
- [ ] Add Whisper model finetuning (future)

## License
MIT