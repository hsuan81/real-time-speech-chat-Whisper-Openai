import os
import queue
import sys
import threading
from time import sleep
stop = threading.Event()
work_queue = queue.Queue()
work_queue.put('Item 1')
work_queue.put('Item 2')


def process():
    """Long lasting operation"""
    # while not stop.is_set() and not work_queue.empty():
    while not stop.is_set():
        for i in range(10):
            sleep(1)
            print("waiting...")
        item = work_queue.get(1)
        # process item
        print(f"Done task: {item}")
        work_queue.task_done()
    if stop.is_set():
        print("KeyboardInterrupt detected, closing background thread. ")
    print("Terminating the thread.")


def main(argv):
    t = threading.Thread(target=process)
    t.start()
    try:
        # while t.is_alive():
        #     print("Waiting for background thread to finish")
        #     sleep(1)
        print("Thread is joined...")
        t.join()
    except KeyboardInterrupt:
        print("alive before:", t.is_alive())
        stop.set()
        print("alive after:", t.is_alive())
        print("Closing main-thread.Please wait for background thread to finish the current item.")
        return 0
    finally:
        print("Running to finally block.")
    # work_queue.join()
    print("Thread finished all tasks, exiting")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))