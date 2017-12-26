import numpy as np, sys, math, os
import time, threading
import atexit, shutil

def red_str(s):
    return '\033[1;31;40m' + s + '\033[0m'

def log(s, red = False, **kw):
    s = str(s)
    if red:
        s = red_str(s)
    print(s, **kw)

class Timer:
    def loop(self):
        while not self.end:
            time.sleep(0.1)
            print('\r%.1fs' % (time.time() - self.start_time), end = '')

    def start(self):
        self.start_time = time.time()
        self.end = False
        self.thread = threading.Thread(target = self.loop)
        self.thread.setDaemon(True)
        self.thread.start()

    def stop(self):
        self.end = True
        self.thread.join()
        print('\r                \r', end = '')
        return time.time() - self.start_time
_timer = Timer()

def main():
    print('hello world, log')
    log(10)

if __name__ == '__main__':
    main()

