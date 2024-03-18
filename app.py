

def main():
    # This queue holds all the 1-second audio chunks
    audio_queue = queue.Queue()

    stop = False
    while not stop:
        try:
            thread1.get_audio()
            thread2.transcribe()
            thread3.collect_text()
            stop_listen_to_user()
            bot_thread1.get_bot_response(user_text)
            bot_thread2.output_bot_voice(bot_answer)
            resume_listen_to_user() 
        except KeyboardInterrupt:
            stop = True   