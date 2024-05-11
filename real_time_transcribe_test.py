#! python3.7

import argparse
import io
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
import speech_recognition as sr
import webrtcvad
from dotenv import load_dotenv

# import whisper
from faster_whisper import WhisperModel
from openai import OpenAI

# import torch


WHISPER_LANGUAGE = "en"
WHISPER_THREADS = 4
LENGHT_IN_SEC: int = 5
# Visualization (expected max number of characters for LENGHT_IN_SEC audio)
MAX_SENTENCE_CHARACTERS = 80
# transcribe = None
end_of_speech = None
silence_count = 0


def source_from_microphone(
    audio_queue, default_microphone=None, energy_threshold=None, record_timeout=None
):
    # global transcribe
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
                    # source = sr.Microphone(sample_rate=16000, device_index=index)
                    break
    else:
        source = sr.Microphone(sample_rate=16000)

    with source:
        recorder.adjust_for_ambient_noise(source)

    def record_callback(_, audio: sr.AudioData) -> None:
        """
        Threaded callback function to receive audio data when recordings finish.
        audio: An AudioData containing the recorded bytes.
        """
        # global transcribe
        # Grab the raw bytes and push it into the thread safe queue.
        # global transcribe
        # print("Callback starts...")
        data = audio.get_raw_data()
        audio_queue.put(data)
        # with transcribe_lock:
        #     transcribe = True
        transcribe.set()

    # Create a background thread that will pass us raw audio bytes.
    # We could do this manually but SpeechRecognizer provides a nice helper.
    stop_listening = recorder.listen_in_background(
        source, record_callback, phrase_time_limit=record_timeout
    )

    # Cue the user that we're ready to go.
    print("Model loaded.\n")

    return stop_listening


def source_from_multiaudiofiles(
    audio_path_list, silence_duration, slice_length_sec, audio_queue, stop_event
):
    """
    Slice the audiofile (.wav file) with fixed interval specified and put the raw data to queue.
    The aim of this function is to emulate the capture from microphone and data created by using Speech Recognition library.
    """
    frame_rate = 16000
    # print(frame_rate)
    frames_per_slice = int(frame_rate * slice_length_sec)
    channels = 1
    # print(channels)
    sample_width = 2
    for file_path in audio_path_list:
        with wave.open(file_path, "rb") as wf:
            # Retrieve the sample rate of the audio file, and then break it down into 1-sec long data for processing
            # frame_rate = wf.getframerate()
            # print(frame_rate)
            # frames_per_slice = int(frame_rate * slice_length_sec)
            # channels = wf.getnchannels()
            # print(channels)
            # sample_width = wf.getsampwidth()
            # print(sample_width)
            offset = None
            # vad = webrtcvad.Vad(1)
            # frame_duration = 30
            # frame_length = int(frame_rate * frame_duration / 1000) * sample_width * channels
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
        # print("---sleep---")
        sleep(silence_duration)
    # Signal that slicing is done
    # audio_queue.put(None)
    print()
    # print("start, end time:", start_time, end_time)
    # print("reading audio duration:", end_time - start_time)
    # print("Audio file loaded.")
    # print("Finished slicing audio file.")


def source_from_audiofile(file_path, slice_length_sec, audio_queue, stop_event):
    """
    Slice the audiofile (.wav file) with fixed interval specified and put the raw data to queue.
    The aim of this function is to emulate the capture from microphone and data created by using Speech Recognition library.
    """
    with wave.open(file_path, "rb") as wf:
        # Retrieve the sample rate of the audio file, and then break it down into 1-sec long data for processing
        frame_rate = wf.getframerate()
        frames_per_slice = int(frame_rate * slice_length_sec)
        channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        offset = None
        # vad = webrtcvad.Vad(1)
        # frame_duration = 30
        # frame_length = int(frame_rate * frame_duration / 1000) * sample_width * channels

        while not stop_event.is_set():
            frames = wf.readframes(frames_per_slice)
            if len(frames) == 0:  # Check for end of file
                break
            audio_data = sr.AudioData(frames, frame_rate, sample_width)
            audio_queue.put(audio_data.get_raw_data())
            transcribe.set()

        # Signal that slicing is done
        # audio_queue.put(None)
        print()
        print("Audio file loaded.")
        # print("Finished slicing audio file.")


def main():
    global transcribe, end_of_speech
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="medium",
        help="Model to use",
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
    parser.add_argument(
        "--phrase_timeout",
        default=3,
        help="How much empty space between recordings before we "
        "consider it a new line in the transcription.",
        type=float,
    )
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

    # The last time a recording was retrieved from the queue.
    # phrase_time = None

    # We use SpeechRecognizer to record our audio because it has a nice feature where it can detect when speech ends.
    # recorder = sr.Recognizer()
    # recorder.energy_threshold = args.energy_threshold
    # Definitely do this, dynamic energy compensation lowers the energy threshold dramatically to a point where the SpeechRecognizer never stops recording.
    # recorder.dynamic_energy_threshold = False

    # print(sr.Microphone.list_microphone_names())
    # print(sr.Microphone.list_working_microphones())

    # Important for linux users.
    # Prevents permanent application hang and crash by using the wrong Microphone
    # if "linux" in platform:
    #     mic_name = args.default_microphone
    #     if not mic_name or mic_name == "list":
    #         print("Available microphone devices are: ")
    #         for index, name in enumerate(sr.Microphone.list_microphone_names()):
    #             print(f'Microphone with name "{name}" found')
    #         return
    #     else:
    #         for index, name in enumerate(sr.Microphone.list_microphone_names()):
    #             if mic_name in name:
    #                 source = sr.Microphone(sample_rate=16000, device_index=index)
    #                 break
    # else:
    #     source = sr.Microphone(sample_rate=16000)

    # Load / Download model
    model = args.model
    if args.model != "large" and not args.non_english:
        model = model + ".en"
    # audio_model = whisper.load_model(model)
    audio_model = WhisperModel(
        "tiny",
        device="cpu",
        compute_type="int8",
        cpu_threads=WHISPER_THREADS,
        download_root="./models",
    )

    record_timeout = args.record_timeout
    phrase_timeout = args.phrase_timeout
    # Thread safe Queue for passing data from the threaded recording callback.
    audio_queue = Queue()
    length_queue = Queue(maxsize=LENGHT_IN_SEC)
    transcription_queue = Queue()

    # with source:
    #     recorder.adjust_for_ambient_noise(source)

    # def record_callback(_, audio: sr.AudioData, transcribe=transcribe) -> None:
    #     """
    #     Threaded callback function to receive audio data when recordings finish.
    #     audio: An AudioData containing the recorded bytes.
    #     """
    #     # Grab the raw bytes and push it into the thread safe queue.
    #     # global transcribe
    #     # print("Callback starts...")
    #     data = audio.get_raw_data()
    #     audio_queue.put(data)
    #     # with transcribe_lock:
    #     #     transcribe = True
    #     transcribe.set()

    # # Create a background thread that will pass us raw audio bytes.
    # # We could do this manually but SpeechRecognizer provides a nice helper.
    # stop_listening = recorder.listen_in_background(
    #     source, record_callback, phrase_time_limit=record_timeout
    # )

    # # Cue the user that we're ready to go.
    # print("Model loaded.\n")

    def consumer_thread(transcribe: threading.Event, thread_terminate: threading.Event):
        # global stored_transcription
        transcription = ""
        pause_count = 0
        # with transcribe_lock:
        #     transcribe = True
        transcribe.set()
        current_openaithread_id = None
        openaithread = None
        stored_transcription = []

        while True:
            # if transcribe.is_set():
            if length_queue.qsize() >= LENGHT_IN_SEC:
                with length_queue.mutex:
                    length_queue.queue.clear()
                    # if transcription != '':
                    # print()
                    # print("-" + transcription + "-")
                    stored_transcription.append(transcription)
                    transcription = ""
                    print()
            if thread_terminate.is_set():
                print("Stopping the main thread and signal to stop subthread...")
                # if current_openaithread_id is not None:
                # print("joining subthread")
                # openaithread.join()
                break
            if current_openaithread_id is None:
                openaithread = threading.Thread(
                    target=chat_thread, args=(transcribe, thread_terminate)
                )
                openaithread.start()
                current_openaithread_id = openaithread.native_id
                print(f"Current openaithread id: {current_openaithread_id}")
                prepare_start_time = time.time()

            # print("Prepare time:", prepare_end_time - prepare_start_time)

            try:
                transcription_start_time = time.time()
                audio_data = audio_queue.get(timeout=2)
                length_queue.put(audio_data)
                if not transcribe.is_set():
                    transcribe.set()
                # if transcription_queue.qsize() == 0:
                #     stored_transcription = []

                audio_data_to_process = b""
                for i in range(length_queue.qsize()):
                    # We index it so it won't get removed
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
                if transcription == "":
                    # print("\n no text")
                    pause_count += 1
                else:
                    print(transcription, end="\r", flush=True)
                    # print(transcription+"-")
                    pause_count = 0
                # print(
                #     "Transcribing duration: ",
                #     transcription_end_time - transcription_start_time,
                # )

            except queue.Empty:
                #     # print("2-2")
                print()
                # print("queue is empty", audio_queue.empty(), length_queue.empty())
                pause_count += 1

            if pause_count >= 1:
                prepare_end_time = time.time()

                if transcribe.is_set():
                    # transcribe.clear()
                    print("Pause, Transcribe:", transcribe.is_set())
                    transcribe.clear()
                    if transcription != "":
                        # print("add empty 2")
                        # print()
                        # print("1-" + transcription + "-")
                        stored_transcription.append(transcription)
                        # transcription = ""
                    audio_data_to_process = b""
                    while not audio_queue.empty():
                        # We index it so it won't get removed
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
                        print()
                        print("2-" + transcription + "-")
                        print(stored_transcription)
                    transcription_queue.put("".join(stored_transcription))
                    with length_queue.mutex:
                        length_queue.queue.clear()
                        # print("Length:", len(transcription))
                    stored_transcription = []
                    # transcribe.clear()
                    # sleep(1)

            else:
                if not transcribe.is_set():
                    print("Transcribing...")
                    transcribe.set()

                continue

        # print()
        # print("size of audio_queue", audio_queue.qsize())
        print("Terminating?", thread_terminate.is_set())
        if thread_terminate.is_set():
            # print("audio queue empty?", audio_queue.empty())
            if transcription != "":
                stored_transcription.append(transcription)
            audio_data_to_process = b""
            while not audio_queue.empty():
                # We index it so it won't get removed
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
            # print()
            # print("Final transcription",transcription)
            if transcription != "":
                stored_transcription.append(transcription)
            transcription_queue.put("".join(stored_transcription))
            print(stored_transcription)
            if current_openaithread_id is not None:
                print("Join the subthread...")
                openaithread.join()
        # print("Transcribing start, end time:", prepare_start_time, prepare_end_time)
        # print("Transcribing duration:", abs(prepare_start_time - prepare_end_time))
        print("consumer thread is closed")

    def chat_thread(transcribe: threading.Event, thread_terminate: threading.Event):
        # global transcribe
        # print("Chat thread is running...")
        response_queue = Queue()
        load_dotenv()  # Loads the .env file and sets its contents as environment variables
        api_key = os.getenv("OPENAI_API_KEY")
        client = OpenAI()
        messages = [
            {
                "role": "system",
                "content": "You are a well-rounded assistant, skilled in answering all kinds of questions and offering useful advice to people.",
            },
        ]

        def form_message_from_transcription(transcription: str) -> None:
            messages.append({"role": "user", "content": transcription})

        def form_message_from_response(response: str) -> None:
            messages.append({"role": "assistant", "content": response})

        def response_handler():
            while not response_queue.empty() and not transcribe.is_set():
                response = response_queue.get()
                print("AI Response:", response)

        # print("*** Finished setting up chat ***")
        # print("transcription queue size:", transcription_queue.qsize())
        # print("Transcribe:", transcribe.is_set())

        while True:
            # print("chat_thread loop start")
            sleep(1)
            if thread_terminate.is_set():
                break
            if transcribe.is_set():
                # print("Waiting for user transcription...")
                continue
            # print("Subworking...")
            try:
                # print("transcription_queue:", transcription_queue.qsize())
                if transcription_queue.qsize() > 0:
                    user_input = transcription_queue.get(timeout=1)
                    form_message_from_transcription(user_input)
                    # print("chat prepared")
                    response = client.chat.completions.create(
                        model="gpt-3.5-turbo", messages=messages
                    )
                    print("get response")
                    extracted_response = response.choices[0]
                    response_queue.put(extracted_response.message.content)
                    # if not transcribe.is_set():
                    #     print("Response status:", extracted_response.finish_reason)
                    #     print("AI response: ", extracted_response.message.content)
                    #     form_message_from_response(extracted_response.message.content)
                    # print("Getting user input:", user_input)
                    transcription_queue.task_done()
            except Exception as e:
                print("Exception occurs:", e)
                continue
            else:
                response_handler()
        print("Subthread is terminated.")

        # if consumer_thread_terminate.is_set():
        #    print(messages)

        # audio_queue.task_done()

    # monitor = threading.Thread(target=monitor_thread, daemon=True)
    # consumer = threading.Thread(target=consumer_thread, daemon=True)
    transcribe = threading.Event()
    # send_data = threading.Event()
    thread_terminate = threading.Event()
    stop_listening = None
    audio_thread = None
    # recorder = sr.Recognizer()
    stop_listening = None
    mode = args.mode
    if mode == "test":
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
    # monitor.start()
    if mode == "test":
        audio_thread.start()
    consumer.start()

    try:
        # print("1. running?", consumer.is_alive())
        while consumer.is_alive():
            # sleep for a few secs to let process print out transcription here
            sleep(3)
            continue
            # if not transcribe:
            # transcription_queue.join()
            # consumer_thread_terminate.set()

        # chat.join()

    except KeyboardInterrupt:
        print("KeyboardInterrupt occurs, running?", consumer.is_alive())
        if mode == "prod":
            stop_listening(wait_for_stop=False)
        thread_terminate.set()
        consumer.join()
        # if mode == "test":

        # transcription_queue.join()

        print()

    finally:
        print()
        while consumer.is_alive():
            print("Child thread is still running...")
            sleep(3)
        # print("Child thread is terminated.")
        print("Exiting...")
    print("outside finally")

    # while True:
    #     try:
    #         now = datetime.utcnow()
    #         # Pull raw recorded audio from the queue.
    #         if not audio_queue.empty():
    #             phrase_complete = False
    #             # If enough time has passed between recordings, consider the phrase complete.
    #             # Clear the current working audio buffer to start over with the new data.
    #             if phrase_time and now - phrase_time > timedelta(seconds=phrase_timeout):
    #                 phrase_complete = True
    #             # This is the last time we received new audio data from the queue.
    #             phrase_time = now

    #             # Combine audio data from queue
    #             audio_data = b''.join(audio_queue.queue)
    #             audio_queue.queue.clear()

    #             # Convert in-ram buffer to something the model can use directly without needing a temp file.
    #             # Convert data from 16 bit wide integers to floating point with a width of 32 bits.
    #             # Clamp the audio stream frequency to a PCM wavelength compatible default of 32768hz max.
    #             audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

    #             # Read the transcription.
    #             print("/* start transcribing */")
    #             # result = audio_model.transcribe(audio_np, fp16=torch.cuda.is_available())

    #             segments, _ = audio_model.transcribe(audio_np, language=WHISPER_LANGUAGE, beam_size=5)
    #             text = " ".join([s.text for s in segments])
    #             # text = result['text'].strip()

    #             # If we detected a pause between recordings, add a new item to our transcription.
    #             # Otherwise edit the existing one.
    #             # if phrase_complete:
    #             transcription.append(text)
    #             # else:
    #                 # transcription[-1] = text

    #             # Clear the console to reprint the updated transcription.
    #             os.system('cls' if os.name=='nt' else 'clear')
    #             for line in transcription:
    #                 print(line)
    #             # Flush stdout.
    #             print('', end='', flush=True)

    #             # Infinite loops are bad for processors, must sleep.
    #             sleep(0.25)
    #     except KeyboardInterrupt:
    #         break

    # print("\n\nTranscription:")
    # for line in transcription:
    #     print(line)


if __name__ == "__main__":
    main()
