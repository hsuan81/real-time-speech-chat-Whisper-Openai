#! python3.7

import argparse
import os
import time
import re
import numpy as np
import speech_recognition as sr
# import whisper
from faster_whisper import WhisperModel
# import torch

from datetime import datetime, timedelta
from queue import Queue
from time import sleep
from sys import platform
import threading
import queue
from openai import OpenAI
from dotenv import load_dotenv

WHISPER_LANGUAGE = "en"
WHISPER_THREADS = 4
LENGHT_IN_SEC: int = 5
# Visualization (expected max number of characters for LENGHT_IN_SEC audio)
MAX_SENTENCE_CHARACTERS = 80
transcribe = None
end_of_speech = None
silence_count = 0


def main():
    global transcribe, end_of_speech
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="medium", help="Model to use",
                        choices=["tiny", "base", "small", "medium", "large"])
    parser.add_argument("--non_english", action='store_true',
                        help="Don't use the english model.")
    parser.add_argument("--energy_threshold", default=1000,
                        help="Energy level for mic to detect.", type=int)
    parser.add_argument("--record_timeout", default=2,
                        help="How real time the recording is in seconds.", type=float)
    parser.add_argument("--phrase_timeout", default=3,
                        help="How much empty space between recordings before we "
                             "consider it a new line in the transcription.", type=float)
    if 'linux' in platform:
        parser.add_argument("--default_microphone", default='pulse',
                            help="Default microphone name for SpeechRecognition. "
                                 "Run this with 'list' to view available Microphones.", type=str)
    args = parser.parse_args()

    # The last time a recording was retrieved from the queue.
    phrase_time = None
    # Thread safe Queue for passing data from the threaded recording callback.
    audio_queue = Queue()
    # We use SpeechRecognizer to record our audio because it has a nice feature where it can detect when speech ends.
    recorder = sr.Recognizer()
    recorder.energy_threshold = args.energy_threshold
    # Definitely do this, dynamic energy compensation lowers the energy threshold dramatically to a point where the SpeechRecognizer never stops recording.
    recorder.dynamic_energy_threshold = False

    # print(sr.Microphone.list_microphone_names())
    print(sr.Microphone.list_working_microphones())


    # Important for linux users.
    # Prevents permanent application hang and crash by using the wrong Microphone
    if 'linux' in platform:
        mic_name = args.default_microphone
        if not mic_name or mic_name == 'list':
            print("Available microphone devices are: ")
            for index, name in enumerate(sr.Microphone.list_microphone_names()):
                print(f"Microphone with name \"{name}\" found")
            return
        else:
            for index, name in enumerate(sr.Microphone.list_microphone_names()):
                if mic_name in name:
                    source = sr.Microphone(sample_rate=16000, device_index=index)
                    break
    else:
        source = sr.Microphone(sample_rate=16000)



    # Load / Download model
    model = args.model
    if args.model != "large" and not args.non_english:
        model = model + ".en"
    # audio_model = whisper.load_model(model)
    audio_model = WhisperModel("tiny", device="cpu", compute_type="int8", cpu_threads=WHISPER_THREADS, download_root="./models")

    record_timeout = args.record_timeout
    phrase_timeout = args.phrase_timeout
    # transcribe = True
    transcribe_lock = threading.Lock()
    consumer_thread_terminate = threading.Event()
    stored_transcription = []

    with source:
        recorder.adjust_for_ambient_noise(source)

    def record_callback(_, audio:sr.AudioData) -> None:
        """
        Threaded callback function to receive audio data when recordings finish.
        audio: An AudioData containing the recorded bytes.
        """
        # Grab the raw bytes and push it into the thread safe queue.
        # print("/* detecting audio */")
        global transcribe, end_of_speech
        # print("Callback starts...")
        data = audio.get_raw_data()
        audio_queue.put(data)
        # print("Put data into audio_queue")
        # print("record", audio_queue.qsize())
        with transcribe_lock:
            transcribe = True
            # print("setting up transcribe: True")
        
    # Create a background thread that will pass us raw audio bytes.
    # We could do this manually but SpeechRecognizer provides a nice helper.
    stop_listening = recorder.listen_in_background(source, record_callback, phrase_time_limit=record_timeout)

    # Cue the user that we're ready to go.
    print("Model loaded.\n")

    length_queue = Queue(maxsize=LENGHT_IN_SEC)
    transcription_queue = Queue()

    def monitor_thread():
        global end_of_speech, silence_count

        while True:
            starting_audio_queue_length = length_queue.qsize()
            # print(time.ctime())
            time.sleep(2)
            # print(time.ctime())
            ending_audio_queue_length = length_queue.qsize()
            # print("*** transcribe in monitor ***:", transcribe)
            # print("*** start size:", starting_audio_queue_length)
            # print("*** end size:", ending_audio_queue_length)

            if starting_audio_queue_length == ending_audio_queue_length:
                silence_count += 1
            else:
                silence_count = 0
            if silence_count >= 2:
                end_of_speech = True
                print("*** setting up end of speech ...", end_of_speech)
            


    def consumer_thread():
        global transcribe
        transcription = ''
        pause_count = 0
        with transcribe_lock:
            transcribe = True
        

        while True:
            # if not consumer_thread_terminate.is_set() and transcribe:
        # while not consumer_thread_terminate.is_set() and transcribe:
            # if transcribe:
                if length_queue.qsize() >= LENGHT_IN_SEC:
                    with length_queue.mutex:
                        length_queue.queue.clear()
                        if transcription != '': stored_transcription.append(transcription)
                        print()
                # print("1",audio_queue.qsize())
                # print("transcribe-1", transcribe)
                try:
                    # print("Beginning:", transcribe)

                    transcription_start_time = time.time()
                    audio_data = audio_queue.get(timeout=3)                    
                    length_queue.put(audio_data)

                    audio_data_to_process = b""
                    for i in range(length_queue.qsize()):
                        # We index it so it won't get removed
                        audio_data_to_process += length_queue.queue[i]
                    
                    audio_np = np.frombuffer(audio_data_to_process, dtype=np.int16).astype(np.float32) / 32768.0
                    
                    segments, _ = audio_model.transcribe(audio_np, language=WHISPER_LANGUAGE, beam_size=5)

                    transcription = ''
                    for s in segments:
                        transcription += s.text 
                    

                    transcription = re.sub(r"\s\s+", "", transcription)
                    

                    # transcription_postprocessing_end_time = time.time()

                    print(transcription, end="\r", flush=True)
                    # print(transcribe)

                
                
                except queue.Empty:
                    # print("2-2")
                    transcription_end_time = time.time()
                    # print()
                    # print("queue is empty", audio_queue.empty(), length_queue.empty())
                    pause_count += 1
                    # sprint("time duration:", transcription_end_time-transcription_start_time)
                    # print("Empty audio_queue")
                    if pause_count >= 2:
                        with transcribe_lock:
                            transcribe = False
                            print("Transcribe:", transcribe)
                            pause_count = 0
                    else:
                        continue
                # except Exception as e:
                #     print(e)
                
                finally:
                    pass
                    # print("Running consumer finally block")

                    # audio_data_to_process = b""
                    # for i in range(length_queue.qsize()):
                    #     # We index it so it won't get removed
                    #     audio_data_to_process += length_queue.queue[i]
                    
                    # audio_np = np.frombuffer(audio_data_to_process, dtype=np.int16).astype(np.float32) / 32768.0
                    
                    # # print("...transcribing...", end="\r", flush=True)
                    # segments, _ = audio_model.transcribe(audio_np, language=WHISPER_LANGUAGE, beam_size=5)
                    # # print("", end="\r", flush=True)

                    # transcription = ''
                    # for s in segments:
                    #     transcription += s.text 

                    # transcription = re.sub(r"\s\s+", "", transcription)
                    # transcription_postprocessing_end_time = time.time()

                    # print(transcription, end="\r", flush=True)
                    
                #     print("*** transcribe ***", transcribe)
                #     print("*** end of speech ***", end_of_speech)
                # if end_of_speech:
                    # with transcribe_lock:
                    #     transcribe = False
                    # print("setting up transcribe: False")
                    
                    # print("Stoping transcibing")
                
            
        # print()
        # print("size of audio_queue", audio_queue.qsize())
        print("Terminating?", consumer_thread_terminate.is_set())
        if consumer_thread_terminate.is_set():
            print("audio queue empty?", audio_queue.empty())
            if transcription != '':
                stored_transcription.append(transcription)
            audio_data_to_process = b""
            while not audio_queue.empty():
                # We index it so it won't get removed
                audio_data_to_process += audio_queue.get(timeout=1)
            audio_np = np.frombuffer(audio_data_to_process, dtype=np.int16).astype(np.float32) / 32768.0
            segments, _ = audio_model.transcribe(audio_np, language=WHISPER_LANGUAGE, beam_size=5)
            transcription = ''
            for s in segments:
                transcription += s.text
            transcription = re.sub(r"\s\s+", "", transcription)
            # print()
            # print("Final transcription",transcription)
            stored_transcription.append(transcription)
            transcription_queue.put(''.join(stored_transcription))
            print(stored_transcription)

        print('consumer thread is closed')

    def chat_thread():
        # global transcribe
        print("Chat thread is running...")
        load_dotenv()  # Loads the .env file and sets its contents as environment variables
        # api_key = os.getenv('OPENAI_API_KEY')
        client = OpenAI()
        messages = [
            {"role": "system", "content": "You are a well-rounded assistant, skilled in answering all kinds of questions and offering useful advice to people."},
        ]

        def form_message_from_transcription(transcription: str) -> None:
            messages.append({"role": "user", "content": transcription})
        
        def form_message_from_response(response: str) -> None:
            messages.append({"role": "assistant", "content": response})

        print("*** Finished setting up chat ***")
        print("transcription queue size:", transcription_queue.qsize())
        print("Transcribe:", transcribe)
        while True:
            if consumer_thread_terminate.is_set():
                break
            if not transcribe and not transcription_queue.empty():
                print("Pass the condition.")
                user_transcription = transcription_queue.get(timeout=1)
                form_message_from_transcription(user_transcription)

                completion = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
                #   stream=True
                )
                
                response = completion.choices[0].message.content
                print("assistant: ", response)
                form_message_from_response(response)
                print(messages)
                transcription_queue.task_done()
                break
            else:
                continue

            # if consumer_thread_terminate.is_set():
            #    print(messages)
                


                # audio_queue.task_done()
    # monitor = threading.Thread(target=monitor_thread, daemon=True)
    # consumer = threading.Thread(target=consumer_thread, daemon=True)
    consumer = threading.Thread(target=consumer_thread)
    # chat = threading.Thread(target=chat_thread)
    # monitor.start()
    consumer.start()
    # chat.start()

    try:
        # print("1. running?", consumer.is_alive())
        while consumer.is_alive():
            # sleep for a few secs to let process print out transcription here
            # sleep(3)
            # continue
            if not transcribe:
                # transcription_queue.join()
                consumer_thread_terminate.set()

        # chat.join()


    except KeyboardInterrupt:
        print("KeyboardInterrupt occurs, running?", consumer.is_alive())
        stop_listening(wait_for_stop=False)
        consumer_thread_terminate.set()
        consumer.join()
        # transcription_queue.join()

        print()
        
    
    finally:
        print()
        while consumer.is_alive():
            print("Child thread is still running...")
            sleep(3)
        print("Child thread is terminated.")
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
