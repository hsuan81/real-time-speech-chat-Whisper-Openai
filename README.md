
# Real-time Speech Conversations with GPT3.5 (Multithreading)

This project achieves a similar feat as the OpenAI's Chatgpt 4o real-time speech conversation feature using multithreading, allowing users to have natural conversations with an AI in real-time!


## Demo

coming soon!!


## Features

- **Real-Time Interaction**: Experience seamless, natural conversations with GPT3.5, just like chatting with a friend.
- **Interrupt GPT's saying anytime by talking**: You can interrupt GPT's speech anytime you'd like by talking anything, just like what you'll do in a conversation with real human!!
- **Multithreading Architecture**: Leveraging sophisticated multithreading architecture to implement the real-time smooth and responsive dialogues. An alternative to OpenAI's real-time voice conversation feature.

## ⚠️ Note

This project is a demonstration of the possibility of implementing real-time conversation with AI using multithreading. It is not yet commercial-ready and may contain bugs and limitations. The primary goal is to showcase the potential and provide a foundation for further development.

### Known Issues and Limitations:
- **Bugs**: Errors of reading already closed audio file occur sometimes, but it won’t stop the program.

- **Better Know before Running**: 
  - Using earphones is a must to try this application because it keeps listening all the time!!! By using the earphones, sound from your pc will not be treated as your voice and accidentally interrupt OpenAI's speech.
  - Transcribing starts once the model being loaded. Keep silent before hearing the OpenAI's speech.
  - Run the application and talk at a rather quiet place. The vad feature is not optimized. Talking at a rather noisy place is prone to transcribing falsely.
  - DeprecationWarning `.with_streaming_response.method()` of OpenAI tts may occur. It’s an old problem and it won’t affect the execution of the program.

- **Performance**: The application might not handle conversations efficiently. You may experience some latency in conversations.


Contributions and feedback are welcome! Let's work together to improve and expand the capabilities of real-time speech AI interactions.

## Tech Stack

- [Speech Recognition](https://github.com/Uberi/speech_recognition)
- [Fast-Whisper](https://github.com/SYSTRAN/faster-whisper)
- OpenAI Whisper and TTS

Please see `requirements.txt` for details.


## Installation

Install required packages locally

```bash
  pip install -r requirements.txt
```

 Or start a conda env with required packages
```bash
  conda create --name myenv pip python=3.10
  pip install -r requirements.txt
```

For MacOS, you need to install portaudio with homebrew before installing pyaudio
```bash
brew install portaudio
```



## Run

Before running, you need to specify your own OPENAI KEY in .env. See `.env.example` to set up your .env file.

There are two modes, "prod" and "test". Specify which mode you'd like to run.

If you'd like to chat in real-time, use "prod" mode.

```bash
  python real_time_voice_conversation_openai.py --mode "prod"
```

In "test" mode, you need to prepare wav files in advance and set up the metadata as follows and specify the location of the file as `source_from_multiaudiofiles()` input:

- channels: 1 
- sample rates: 16000
- precision: 16-bit




## License

[MIT License](https://choosealicense.com/licenses/mit/)


## Reference

This project is built on and references the following projects.

[
whisper-live-transcription](https://github.com/gaborvecsei/whisper-live-transcription)



