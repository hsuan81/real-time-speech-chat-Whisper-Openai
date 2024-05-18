import os
import time
import wave

import pyaudio
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()  # Loads the .env file and sets its contents as environment variables

api_key = os.getenv("OPENAI_API_KEY")


client = OpenAI()

response = client.audio.speech.create(
    model="tts-1",
    voice="alloy",
    input="The vast amount of information enriches your life and at the same time divides your life into countless small segments. At the same time, as a person grows up, you are responsible for yourself, for your work, for your partner, for your children, for your parents.",
    response_format="wav",
)

response.stream_to_file("output.wav")

# Open the wave file
wf = wave.open("output.wav", "rb")

# Instantiate PyAudio
p = pyaudio.PyAudio()


# Define callback function to stream audio
def callback(in_data, frame_count, time_info, status):
    # if status:
    #     return (None, pyaudio.paComplete)
    data = wf.readframes(frame_count)
    return (data, pyaudio.paContinue)


# Open a stream with the callback function
stream = p.open(
    format=p.get_format_from_width(wf.getsampwidth()),
    channels=wf.getnchannels(),
    rate=wf.getframerate(),
    output=True,
    stream_callback=callback,
)

# Start the stream
stream.start_stream()

# Wait for stream to finish (4)
while stream.is_active():
    time.sleep(0.01)
    print("playing")

# Keep the stream active until it is stopped
# while stream.is_active():
#     if stop_event.is_set():
#         stream.stop_stream()
#         break

# Close the stream and PyAudio
stream.close()
wf.close()
p.terminate()
