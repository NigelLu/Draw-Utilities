"""Main Drawing Module main.py
"""
import time
from config import config as cfg


def main():
    pass


if __name__ == "__main__":
    start_time = time.time()
    print(
        f">>> Main process started at {time.strftime('%H:%M:%S, %m/%d/%Y',time.localtime(start_time))} local time")
    main()
    end_time = time.time()
    print(f">>> Main process finished in {end_time - start_time} seconds")
    print(
        f">>> Main process finished at {time.strftime('%H:%M:%S, %m/%d/%Y',time.localtime(end_time))} local time")
