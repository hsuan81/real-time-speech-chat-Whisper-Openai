#! python3.10

import argparse
import os
import queue
import re
import threading
import time
import wave
from datetime import datetime, timedelta
from queue import Queue
from sys import platform
from time import sleep

import numpy as np
import pyaudio
import speech_recognition as sr
from dotenv import load_dotenv
from faster_whisper import WhisperModel
from openai import OpenAI

WHISPER_LANGUAGE = "en"
WHISPER_THREADS = 4
LENGHT_IN_SEC: int = 5
# Visualization (expected max number of characters for LENGHT_IN_SEC audio)
MAX_SENTENCE_CHARACTERS = 80
# transcribe = None


def source_from_microphone(
    audio_queue: Queue,
    default_microphone=None,
    energy_threshold=None,
    record_timeout=None,
):
    recorder = sr.Recognizer()
    recorder.energy_threshold = energy_threshold
    recorder.dynamic_energy_threshold = False

    if "linux" in platform:
        mic_name = default_microphone
        if not mic_name or mic_name == "list":
            print("Available microphone devices are: ")
            for index, name in enumerate(sr.Microphone.list_microphone_names()):
                print(f'Microphone with name "{name}" found')
            return
        else:
            for index, name in enumerate(sr.Microphone.list_microphone_names()):
                if mic_name in name:
                    break
    else:
        source = sr.Microphone(sample_rate=16000)

    with source:
        recorder.adjust_for_ambient_noise(source, duration=1.5)

    def record_callback(_, audio: sr.AudioData) -> None:
        """
        Threaded callback function to receive audio data when recordings finish.
        audio: An AudioData containing the recorded bytes.
        """
        data = audio.get_raw_data()
        audio_queue.put(data)
        # set up transcribe event indicating it's in the transcribing mode
        transcribe.set()

    # Cue the user that we're ready to go.
    print("==========Model loaded.==========")
    print(
        "==========IMPORTANT NOTICE: PLEASE USE HEADSETS OR EARPHONES AT A VERY QUIET PLACE.=========="
    )
    print()

    # Create a background thread that will pass us raw audio bytes.
    # We could do this manually but SpeechRecognizer provides a nice helper.
    stop_listening = recorder.listen_in_background(
        source, record_callback, phrase_time_limit=record_timeout
    )
    print(
        "==========Please wait for the speech of the OpenAI and start talk after hearing it.=========="
    )

    return stop_listening


def source_from_multiaudiofiles(
    audio_path_list: list,
    silence_duration: int,
    slice_length_sec: int,
    audio_queue: Queue,
    stop_event: threading.Event,
):
    """
    Pass in a list of audio files and read each audiofile (.wav file) with fixed interval specified and put the raw data to queue.
    Between each audiofile, you cand set up silence duration to emulate stopping in real conversation.
    The aim of this function is to emulate the capture from microphone and data created by using Speech Recognition library.
    Mainly for testing purpose.
    """
    frame_rate = 16000
    frames_per_slice = int(frame_rate * slice_length_sec)
    sample_width = 2
    for file_path in audio_path_list:
        with wave.open(file_path, "rb") as wf:
            start_time = time.time()
            while not stop_event.is_set():
                frames = wf.readframes(frames_per_slice)
                if len(frames) == 0:  # Check for end of file
                    break
                audio_data = sr.AudioData(frames, frame_rate, sample_width)
                sleep(0.5)
                audio_queue.put(audio_data.get_raw_data())
                transcribe.set()
        end_time = time.time()
        sleep(silence_duration)
    print()
    # print("start, end time:", start_time, end_time)
    # print("reading audio duration:", end_time - start_time)
    # print("Audio file loaded.")


def source_from_audiofile(
    file_path: str,
    slice_length_sec: int,
    audio_queue: Queue,
    stop_event: threading.Event,
):
    """
    Slice the audiofile (.wav file) with fixed interval specified and put the raw data to queue.
    The aim of this function is to emulate the capture from microphone and data created by using Speech Recognition library.
    Mainly for testing purpose.
    """
    with wave.open(file_path, "rb") as wf:
        # Retrieve the sample rate of the audio file, and then break it down into 1-sec long data for processing
        frame_rate = wf.getframerate()
        frames_per_slice = int(frame_rate * slice_length_sec)
        sample_width = wf.getsampwidth()

        while not stop_event.is_set():
            frames = wf.readframes(frames_per_slice)
            if len(frames) == 0:  # Check for end of file
                break
            audio_data = sr.AudioData(frames, frame_rate, sample_width)
            audio_queue.put(audio_data.get_raw_data())
            transcribe.set()

        print()
        print("Audio file loaded.")


def main():
    global transcribe
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="tiny",
        help="Model size to use",
        choices=["tiny", "base", "small", "medium", "large"],
    )
    parser.add_argument(
        "--non_english", action="store_true", help="Don't use the english model."
    )
    parser.add_argument(
        "--energy_threshold",
        default=1000,
        help="Energy level for mic to detect.",
        type=int,
    )
    parser.add_argument(
        "--record_timeout",
        default=2,
        help="How real time the recording is in seconds.",
        type=float,
    )
    # parser.add_argument(
    #     "--phrase_timeout",
    #     default=3,
    #     help="How much empty space between recordings before we "
    #     "consider it a new line in the transcription.",
    #     type=float,
    # )
    parser.add_argument(
        "--mode",
        default="test",
        help="Specify the running mode. The 'test' mode take an .wav file as input, "
        "while the 'prod' mode source the audio from microphone",
        choices=["test", "prod"],
    )

    if "linux" in platform:
        parser.add_argument(
            "--default_microphone",
            default="pulse",
            help="Default microphone name for SpeechRecognition. "
            "Run this with 'list' to view available Microphones.",
            type=str,
        )
    args = parser.parse_args()

    # Load / Download model
    model = args.model
    if args.model != "large" and not args.non_english:
        model = model + ".en"
    audio_model = WhisperModel(
        # "tiny",
        model,
        device="cpu",
        compute_type="int8",
        cpu_threads=WHISPER_THREADS,
        download_root="./models",
    )

    record_timeout = args.record_timeout
    # phrase_timeout = args.phrase_timeout

    print("==========Settings==========")
    print("Model size: ", model)
    print("Non-English: ", args.non_english)
    print("Energy Threshold: ", args.energy_threshold)
    print("Record Timeout: ", args.record_timeout)
    print("Mode: ", args.mode)
    print("=============================")
    # Thread safe Queue for passing data from the threaded recording callback.
    audio_queue = Queue()
    # Thread safe Queue for passing audio files to transcribe
    length_queue = Queue(maxsize=LENGHT_IN_SEC)
    # Thread safe Queue for passing transcriptions
    transcription_queue = Queue()

    def consumer_thread(transcribe: threading.Event, thread_terminate: threading.Event):
        """
        This thread is for background listening audio input and put the audio raw data into audio_queue for temporary storage first,
        and then process them in a sliding window like method by moving audio data from audio_queue to length_queue one by one.
        """
        transcription = ""
        pause_count = 0  # pause_count is for counting the silence section
        current_openaithread_id = None
        openaithread = None
        stored_transcription = []
        transcribe.set()
        while True:
            # Clear length_queue and add final definitive transcription to a list if it's full
            if length_queue.qsize() >= LENGHT_IN_SEC:
                with length_queue.mutex:
                    length_queue.queue.clear()
                    stored_transcription.append(transcription)
                    transcription = ""
                    print()
            # Stop the thread if the program is terminated
            if thread_terminate.is_set():
                print("Stopping the main thread and signal to stop subthread...")
                break
            # Create a thread to process interaction with openai if it doesn't exist (move out of the forever loop??)
            if current_openaithread_id is None:
                openaithread = threading.Thread(
                    target=chat_thread, args=(transcribe, thread_terminate)
                )
                openaithread.start()
                current_openaithread_id = openaithread.native_id
                # print(f"Current openaithread id: {current_openaithread_id}")
                prepare_start_time = time.time()

            try:
                # transcription_start_time = time.time()
                # If there is no audio data in the queue after waiting for two secs, it triggers queue.Empty exception and counts one silence section.
                audio_data = audio_queue.get(timeout=2)
                length_queue.put(audio_data)
                audio_queue.task_done()
                if not transcribe.is_set():
                    transcribe.set()

                # Form the audio binary data of all audio data in the length_queue
                audio_data_to_process = b""
                for i in range(length_queue.qsize()):
                    audio_data_to_process += length_queue.queue[i]
                transcription_end_time = time.time()

                audio_np = (
                    np.frombuffer(audio_data_to_process, dtype=np.int16).astype(
                        np.float32
                    )
                    / 32768.0
                )

                segments, _ = audio_model.transcribe(
                    audio_np, language=WHISPER_LANGUAGE, beam_size=5
                )

                transcription = ""
                for s in segments:
                    transcription += s.text

                transcription = re.sub(r"\s\s+", "", transcription)

                # transcription_postprocessing_end_time = time.time()
                # Sometimes, it transcribes silence as an empty string. We treat it as one silence section.
                if transcription == "":
                    pause_count += 1
                else:
                    print(transcription, end="\r", flush=True)
                    pause_count = 0

            except queue.Empty:
                pause_count += 1

            # When more than one silence section is detected, it's treated as the end of the user's speech.
            # The transcribe event is reset and all unprocessed audio data in the audio queue will be processed together.
            if pause_count >= 1:
                prepare_end_time = time.time()

                if transcribe.is_set():
                    transcribe.clear()
                    if transcription != "":
                        stored_transcription.append(transcription)
                    audio_data_to_process = b""
                    while not audio_queue.empty():
                        audio_data_to_process += audio_queue.get(timeout=1)
                    audio_np = (
                        np.frombuffer(audio_data_to_process, dtype=np.int16).astype(
                            np.float32
                        )
                        / 32768.0
                    )
                    segments, _ = audio_model.transcribe(
                        audio_np, language=WHISPER_LANGUAGE, beam_size=5
                    )
                    transcription = ""
                    for s in segments:
                        transcription += s.text
                    transcription = re.sub(r"\s\s+", "", transcription)
                    if transcription != "":
                        stored_transcription.append(transcription)
                    transcription_queue.put("".join(stored_transcription))
                    with length_queue.mutex:
                        length_queue.queue.clear()
                    stored_transcription = []

            else:
                # If silence count is 0, start the transcribing mode.
                if not transcribe.is_set():
                    # print("Transcribing...")
                    transcribe.set()

                continue

        # print("Terminating?", thread_terminate.is_set())
        # If the program is terminate, no need to do the processing???
        if thread_terminate.is_set():
            print("===In consumer terminating process===")
            if transcription != "":
                stored_transcription.append(transcription)
            audio_data_to_process = b""
            while not audio_queue.empty():
                audio_data_to_process += audio_queue.get(timeout=1)
            audio_np = (
                np.frombuffer(audio_data_to_process, dtype=np.int16).astype(np.float32)
                / 32768.0
            )
            segments, _ = audio_model.transcribe(
                audio_np, language=WHISPER_LANGUAGE, beam_size=5
            )
            transcription = ""
            for s in segments:
                transcription += s.text
            transcription = re.sub(r"\s\s+", "", transcription)
            if transcription != "":
                stored_transcription.append(transcription)
            transcription_queue.put("".join(stored_transcription))
            # print(stored_transcription)
            # Join the openai thread to ensure this thread will only stop after openai thread is closed
            if current_openaithread_id is not None:
                print("Waiting for openai thread to be closed...")
                openaithread.join()
        # print("Transcribing start, end time:", prepare_start_time, prepare_end_time)
        # print("Transcribing duration:", abs(prepare_start_time - prepare_end_time))
        print("consumer thread is closed")

    def chat_thread(transcribe: threading.Event, thread_terminate: threading.Event):
        """
        This thread is to send transcription to openai api to get response, use openai tts to transform text to speech and play it right away.
        """
        response_queue = (
            Queue()
        )  # response_queue to store response from openai for later tts processing
        tts_queue = Queue()
        load_dotenv()  # Loads the .env file and sets its contents as environment variables
        api_key = os.getenv("OPENAI_API_KEY")
        client = OpenAI()
        # Set up initial openai prompt setting
        messages = [
            {
                "role": "system",
                "content": "You are a well-rounded assistant, skilled in answering all kinds of questions and offering useful advice to people.",
            },
        ]

        # Instantiate PyAudio
        p = pyaudio.PyAudio()
        output_file = "output.wav"

        # Conversations will be saved and passed to openai api for every request
        history = ""

        def form_message_from_transcription(transcription: str) -> None:
            messages.append({"role": "user", "content": transcription})

        def form_message_from_response(response: str) -> None:
            messages.append({"role": "assistant", "content": response})

        def response_handler(history):
            """
            Start to play the tts audio file and stop playing when it's finished.
            """
            wf = wave.open(output_file, "rb")

            def callback(in_data, frame_count, time_info, status):
                if transcribe.is_set() or thread_terminate.is_set():
                    return (None, pyaudio.paComplete)
                try:
                    data = wf.readframes(frame_count)
                except ValueError:
                    data = []
                if len(data) == 0:
                    return (None, pyaudio.paComplete)
                return (data, pyaudio.paContinue)

            while not response_queue.empty() and not transcribe.is_set():
                # while not tts_queue.empty() and not transcribe.is_set():
                # wf = wave.open(output_file, "rb")
                if not tts_queue.empty():
                    response = response_queue.get()
                    tts_queue.get()
                    wf = wave.open(output_file, "rb")

                    stream = p.open(
                        format=p.get_format_from_width(wf.getsampwidth()),
                        channels=wf.getnchannels(),
                        rate=wf.getframerate(),
                        output=True,
                        stream_callback=callback,
                    )
                    stream.start_stream()

                    while stream.is_active():
                        # Sleeping for a short period is necessary to avoid blocking other threads and processing.
                        sleep(0.5)
                        if thread_terminate.is_set():
                            sleep(0.5)
                            stream.stop_stream()
                            print("====Stop playing the audio====")
                            break
                    stream.close()
                    wf.close()
                    history += response
                    # print()
                    # print("AI Response:", response)
                    form_message_from_response(response)

        while True:
            # Since the loop is running forever to send request anytime when transcribe event is not set, it needs to sleep to let consumer thread to process without blocking.
            sleep(1)
            if thread_terminate.is_set():
                # print(history)
                p.terminate()
                break
            if transcribe.is_set():
                continue
            try:
                print()
                # Start to send messages to openai and generate audio response if there is transcription in the transcription_queue
                if transcription_queue.qsize() > 0:
                    user_input = transcription_queue.get(timeout=1)
                    form_message_from_transcription(user_input)
                    response = client.chat.completions.create(
                        model="gpt-3.5-turbo", messages=messages
                    )
                    extracted_response = response.choices[0]
                    # openai tss
                    tts_response = client.audio.speech.create(
                        model="tts-1",
                        voice="alloy",
                        input=extracted_response.message.content,
                        response_format="wav",
                    )

                    # This method is provided by Openai, and it often raises warning.
                    # It's an old problem, but it doesn't affect the program to proceed.
                    tts_response.stream_to_file(output_file)
                    tts_queue.put(1)
                    # wf = wave.open(output_file, "rb")

                    response_queue.put(extracted_response.message.content)
                    transcription_queue.task_done()
                    print("==========OpenAI is answering==========")
                    response_handler(history)
                    response_queue.task_done()

            except Exception as e:
                print("Exception occurs:", e)
                continue

        print("Openai thread is terminated.")
        # audio_queue.task_done()

    # for switching between transcribing mode and getting openai response mode
    transcribe = threading.Event()
    thread_terminate = threading.Event()  # for signaling termination of this program
    stop_listening = None
    audio_thread = None
    stop_listening = None

    # There are two modes and the program will be processed according to the arguement passed from the commandline.
    mode = args.mode
    if mode == "test":
        # Test mode is for testing purpose, and it sets up audio files as source of input.
        print("Mode: test")
        audio_file_path = "./all_Testing_20silence.wav"
        # file_path_1 = "./Testing_AM_conv16000.wav"
        # file_path_2 = "./Testing2_AM_conv16000.wav"
        file_path_1 = "./tmp_audio/rotterdam_buildings_conv16000.wav"
        file_path_2 = "./tmp_audio/What_is_the_capital_conv16000.wav"
        slice_length_sec = 1
        silence_duration = 2
        audio_thread = threading.Thread(
            target=source_from_multiaudiofiles,
            args=(
                [file_path_1, file_path_2],
                silence_duration,
                slice_length_sec,
                audio_queue,
                thread_terminate,
            ),
        )

    elif mode == "prod":
        # Production mode is for real world usage that it sets microphone as the source of input.
        print("Mode: prod")
        if "linux" not in platform:
            stop_listening = source_from_microphone(
                audio_queue, "pulse", args.energy_threshold, record_timeout
            )
        else:
            stop_listening = source_from_microphone(
                audio_queue,
                args.default_microphone,
                args.energy_threshold,
                record_timeout,
            )

    consumer = threading.Thread(
        target=consumer_thread, args=(transcribe, thread_terminate)
    )
    if mode == "test":
        audio_thread.start()
    consumer.start()

    try:
        while consumer.is_alive():
            # sleep for a few secs to let process print out transcription here
            sleep(3)
            continue

    except KeyboardInterrupt:
        print("\nKeyboardInterrupt occurs, threads are running?", consumer.is_alive())
        if mode == "prod":
            stop_listening(wait_for_stop=False)
        thread_terminate.set()
        consumer.join()
        print()

    finally:
        print()
        while consumer.is_alive():
            print("Consumer thread is still running...")
            sleep(3)
        print("Exiting...")
    print("The program stops.")


if __name__ == "__main__":
    main()
