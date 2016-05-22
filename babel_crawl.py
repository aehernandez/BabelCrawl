from __future__ import print_function

from multiprocess import Process, active_children, Pool
from multiprocess import Manager
from threading import BoundedSemaphore as Semaphore

import requests as re
import numpy as np
import random

import cv2
from PIL import Image
from StringIO import StringIO

import uuid
import os
import sys
import shelve
import atexit
import argparse

import time

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

global_session = re.Session()

# `mount` a custom adapter that retries failed connections for HTTP and HTTPS requests.
global_session.mount("http://", re.adapters.HTTPAdapter(max_retries=1))
global_session.mount("https://", re.adapters.HTTPAdapter(max_retries=1))

# prepare request information
url = 'http://babelia.libraryofbabel.info/babelia.cgi'
headers = {"Content-Type": "application/x-www-form-urlencoded"}


def download_image_thread(location_q, image_q, MAX_DL_THREADS=10):
    print("Running Download Image Thread.")

    
    max_processes = MAX_DL_THREADS 
    print("Creating a thread pool of size {} for downloading images...".format(max_processes))
    pool = Pool(processes=max_processes)
    # Allow us to have n processes runnning, and n processes scheduled to run
    # TODO: Manager is not necessary here, but is used to get around the fact
    # that thread-safe objects cannot be passed by reference, they must be
    # inheretence. A more lightweight solution should be found
    workers = Manager().Semaphore(max_processes*2) 

    def async_download(location):
        image = download_image(location)
        image_q.put((location, image), True)
        print("releasing...")
        workers.release()
        
    while True:
        location = location_q.get(True)
        workers.acquire()
        pool.apply_async(async_download, (location,))
        print("image {} | location {}".format(image_q.qsize(),
            location_q.qsize()))

                
def generate_location_thread(location_q, num_bits):
    print("Running Generate Location Thread.")
    while True:
        value = random.getrandbits(num_bits)
        location_q.put(value, True)

    
def classification_thread(image_q, classifiers, image_path, state, state_lock):
    print("Running Classification Thread")
    iteration = 0

    while True:
        (location, image) = image_q.get(True)

        iteration = iteration + 1
        print("Proccesing image {}".format(iteration))

        regions = detect_interest_regions(image, classifiers)

        print("Found {} regions".format(len(regions)))
        if len(regions) > 0:
            # TODO: stronger unique ids
            unique_id = str(uuid.uuid4())
            print("Saving image with unique id {}".format(unique_id))
            image.save(os.path.join(image_path, "{}.png".format(unique_id)))

            # Save map of uuid to image location
            state_lock.acquire()
            state[unique_id] = location
            state_lock.release()
        

def spin_crawl_threads(state, classifiers, MAX_BIT_SIZE, MAX_DL_THREADS, image_path):
    print("Running threads...")
    manager = Manager()

    location_q = manager.Queue(maxsize=16)
    image_q = manager.Queue(maxsize=64)
    state_lock = manager.Lock()

    generate_location = Process(target=generate_location_thread,
                                args=(location_q, MAX_BIT_SIZE),
                                name="generate_location")
    classification = Process(target=classification_thread,
                             args=(image_q, classifiers, image_path,
                                   state, state_lock), name="classification")
    download_image_t = Process(target=download_image_thread,
                             args=(location_q, image_q, MAX_DL_THREADS), name="download_image")

    download_image_t.start()
    classification.start()
    generate_location.start()

    def kill_threads():
        for thread in active_children():
            thread.terminate()

    atexit.register(kill_threads)
    
    download_image_t.join()
    classification.join()
    generate_location.join()

def download_image(value, session=global_session):
    r = session.post(url, data="location={}".format(value), headers=headers)
    r.raise_for_status()
    return Image.open(StringIO(r.content))


def load_classifiers(paths):
    classifiers = []
    for path in paths:
        print("loading classifier {}".format(path))
        if os.path.isfile(path):
            classifiers.append(cv2.CascadeClassifier(path))
        else:
            sys.exit("error: classifier file not found at {}".format(path))

    return classifiers


def show_image(image, title=""):
    cv2.imshow(title, image)
    cv2.waitKey(0)


def display_regions(image, regions):
    for (x, y, w, h) in regions:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.imshow("Regions Found", image)
        cv2.waitKey(0)


def detect_interest_regions(image, classifiers):
    image = np.array(image)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    interest_regions = []

    for classifier in classifiers:
        detected = classifier.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=2,
            minSize=(30, 30),
            flags=cv2.cv.CV_HAAR_SCALE_IMAGE
        )
        interest_regions.extend(detected)

    return interest_regions


def dict_append(d, key, value):
    if d.has_key(key):
        temp = d[key]
        temp.append(value)
        d[key] = temp
    else:
        d[key] = [value]

def ping_babel(value, verbose=True):
    import time
    try:
        start = time.time()
        download_image(value)
        stop = time.time()
        if verbose:
            print("elapsed request time: {}s".format(stop - start))
    except:
        eprint("error: something went wrong when communicating with babel servers")

    return stop - start


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Crawl http://babelia.libraryofbabel.info/ for faces and other interesting features')
    parser.add_argument('--state_path', dest='shelve_path', default='state.crawl', type=str,
                        help='The path to save the persistent state dictionary used by this application')
    parser.add_argument('--image_path', dest='image_path', default='./images', type=str,
                        help='The path to save images which contain interesting regions')
    parser.add_argument('--max', dest='MAX_BIT_SIZE', type=int, default=50000,
                        help='The maximum number of bits to generate for an image location')
    parser.add_argument('--threaded', '-t', dest='threaded',
            action='store_true', default=False)
    parser.add_argument('--ping', '-p', dest='ping', action='store_true',
            default=False, help="Ping the babel serves to test latency")
    parser.add_argument('--max-dl', dest='MAX_DL_THREADS', default=10, type=int)

    args = parser.parse_args()
    MAX_BIT_SIZE = args.MAX_BIT_SIZE
    MAX_DL_THREADS = args.MAX_DL_THREADS
    shelve_path = args.shelve_path
    image_path = args.image_path
    threaded = args.threaded

    if args.ping:
        while True:
            location = random.getrandbits(MAX_BIT_SIZE)
            ping_babel(location)
            time.sleep(2)

    print("Starting babel image crawl...")
    d = shelve.open(shelve_path)

    def on_exit():
        print("Stopping babel image crawl...")
        d.close()

    atexit.register(on_exit)

    iteration = 0

    if d.has_key('seed'):
        print("Seeding from a previous state")
        random.setstate(d['seed'])
    else:
        print("Starting from a new seed")

    # load haarcascade classifiers
    classifier_paths = ['frontalface_alt2.xml', 'frontalface_alt_tree.xml',
                        'frontalface_alt.xml', 'frontalface_default.xml',
                        'fullbody.xml', 'lowerbody.xml', 'profileface.xml',
                        'upperbody.xml']

    # TODO: This should not be hardcoded in, add to argparse
    classifier_paths = map(lambda s: './classifiers/haarcascade_' + s,
                           classifier_paths)
    classifiers = load_classifiers(classifier_paths)


    if threaded:
        spin_crawl_threads(d, classifiers, MAX_BIT_SIZE, MAX_DL_THREADS, image_path)
    else:
        while True:
            location = random.getrandbits(MAX_BIT_SIZE)
            iteration = iteration + 1
            print("Processed {} images from seed".format(iteration))

            try:
                image = download_image(location)
            except re.HTTPError as http_error:
                eprint("an error occured while attempting to download an image from babel")
                eprint(http_error)
                continue

            regions = detect_interest_regions(image, classifiers)

            print("Found {} regions".format(len(regions)))

            if len(regions) > 0:
                unique_id = uuid.uuid4()
                print("Saving image with unique id {}".format(unique_id))
                image.save(os.path.join(image_path, "{}.png".format(unique_id)))

                # Save map of uuid to image location
                if d.has_key('interest'):
                    temp = d['interest']
                    temp[unique_id] = location
                    d['interest'] = temp
                else:
                    d['interest'] = {unique_id: location}

            d['seed'] = random.getstate()


