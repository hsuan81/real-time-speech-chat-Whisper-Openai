import speech_recognition as sr
import pyaudio
import wave

import wave
import speech_recognition as sr
from queue import Queue
import threading
import time


def slice_audio_to_queue(file_path, slice_length_sec, audio_queue, stop_event):
    with wave.open(file_path, "rb") as wf:
        frame_rate = wf.getframerate()
        frames_per_slice = int(frame_rate * slice_length_sec)

        while not stop_event.is_set():
            frames = wf.readframes(frames_per_slice)
            if len(frames) == 0:  # Check for end of file
                break
            audio_data = sr.AudioData(frames, frame_rate, wf.getsampwidth())
            audio_queue.put(audio_data.get_raw_data())

        # Signal that slicing is done
        audio_queue.put(None)


def process_audio_queue(audio_queue, stop_event):
    while not stop_event.is_set():
        audio_data = audio_queue.get()
        if audio_data is None:  # Check for the termination signal
            break
        # Process the audio_data here
        # For example, you might use a speech recognition API
        print("Processing audio slice...")
        time.sleep(1)  # Simulated processing time
    print("Finished audio processing")


def do_other_work(stop_event):
    while not stop_event.is_set():
        # Perform other tasks here
        print("Doing other work...")
        time.sleep(2)  # Simulated work time


# Usage
def main():
    audio_file_path = "./Testing.wav"
    slice_length_sec = 1
    audio_queue = Queue()
    stop_event = threading.Event()

    # Create and start the audio slicing thread
    audio_thread = threading.Thread(
        target=slice_audio_to_queue,
        args=(audio_file_path, slice_length_sec, audio_queue, stop_event),
    )
    audio_thread.start()

    # Create and start the audio processing thread
    processing_thread = threading.Thread(
        target=process_audio_queue, args=(audio_queue, stop_event)
    )
    processing_thread.start()

    # Create and start a thread for other work
    other_work_thread = threading.Thread(target=do_other_work, args=(stop_event,))
    other_work_thread.start()

    # Simulate running for a limited time (e.g., 20 seconds)
    try:
        time.sleep(20)
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()

    # Wait for threads to finish
    audio_thread.join()
    processing_thread.join()
    other_work_thread.join()

    print("All tasks completed.")


if __name__ == "__main__":
    main()
