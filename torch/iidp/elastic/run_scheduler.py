import os, sys

import argparse
import time

import scheduler

SLEEP_TIME = 10


def main(args):
    # Instantiate scheduler.
    sched = scheduler.Scheduler(args.port, args.config_file)

    try:
        while True:
            time.sleep(SLEEP_TIME)
    except KeyboardInterrupt:
        pass
    finally:
        sched.shut_down()


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Elastic IIDP Scheduler Runner')
    parser.add_argument('-p', '--port', type=int, default=40000,
                        help="Scheduler's Port Number")
    parser.add_argument('-c', '--config_file', type=str, required=True,
                        help="Initial configuration JSON file path")
    main(parser.parse_args())