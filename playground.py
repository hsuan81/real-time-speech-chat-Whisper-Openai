import os
import queue
import sys
import threading
from time import sleep

stop = threading.Event()
work_queue = queue.Queue()
next_queue = queue.Queue()
work_queue.put("Item 1")
work_queue.put("Item 2")


def process(event, event2):
    """Long lasting operation"""
    # while not stop.is_set() and not work_queue.empty():
    while True:
        if stop.is_set():
            break
        # event2.wait()
        # if event2.is_set():
        for i in range(3):
            sleep(1)
            print("waiting...", i)
            try:
                item = work_queue.get(timeout=1)
                next_queue.put(item)
                print(f"Done task: {item}")
                work_queue.task_done()
            # process item
            except queue.Empty:
                print("Queue is empty.")
                event.set()
                break
            finally:
                print("Finished process.")
            # else:
            print("After finally block")

        print("Stop the process thread")
        break

    if stop.is_set():
        print("KeyboardInterrupt detected, closing background thread. ")
    print("Terminating the thread.")


def thread2():
    format_list = []
    while True:
        if work_queue.empty():
            print("Moving...")
            work = next_queue.get()
            next_queue.task_done()
            print("Done moving: {work}")
            if not next_queue.empty():
                continue
            else:
                print("Moving task finished.")
                break
    print("Works have been moved to the final destination.")


def thread3(event, event2):
    while True:
        event.wait()
        print("The event is:", event.is_set())
        if event.is_set():
            print("Event is set.")
            sleep(2)
            event2.set()
            event.cleared()
            break
        else:
            print("Event is not set")
            sleep(2)
        if stop.is_set():
            break
    print("Thread3 is terminated.")


def main(argv):
    event = threading.Event()
    event2 = threading.Event()
    t = threading.Thread(target=process, args=(event, event2), daemon=True)
    # t2 = threading.Thread(target=thread2)
    t3 = threading.Thread(target=thread3, args=(event, event2))

    event2.set()
    t.start()
    # t2.start()
    # t3.start()
    try:
        while t.is_alive():
            # if event.is_set():
            #     print("Event is set")
            #     event.clear()
            #     print("Event is cleared.")
            print("Waiting for background thread to finish")
            sleep(3)
            # event.set()
            # work_queue.join()
            # if work_queue.empty():
            #     stop.set()
            #     print("stop:", stop.is_set())
            #     print("work_queue empty?", work_queue.empty())

            #     # next_queue.join()
            #     break

        # print("Thread is joined...")
        # t.join()
    except KeyboardInterrupt:
        print("alive before:", t.is_alive())
        stop.set()
        # t3.join()
        print("alive after:", t.is_alive())
        print(
            "Closing main-thread.Please wait for background thread to finish the current item."
        )
        return 0
    finally:
        print("Running to finally block.")
        # next_queue.join()
        # t3.join()
    print("Thread finished all tasks, exiting")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
