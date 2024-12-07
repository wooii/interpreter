# Interpreter
i am trying to build a real time transcription tool using python and openai's whisper,
however here is a major problem to solve. my approach is record a set duration of audio and then
send it to openai's api or a local model, however this may cause cut off of the speech and
eventually lose information. for words that are not common, add translation to it.


## Installation
1. Open Terminal and navigate to the directory containing 'pyproject.toml' using the 'cd' command.
2. To install in development mode, run: 'pip install -e .'.
