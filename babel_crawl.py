from __future__ import print_function
from multiprocessing import Process, Lock, active_children
from multiprocessing import JoinableQueue as Queue

from requests_futures.sessions import FuturesSession

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

session = re.Session()

# `mount` a custom adapter that retries failed connections for HTTP and HTTPS requests.
session.mount("http://", re.adapters.HTTPAdapter(max_retries=1))
session.mount("https://", re.adapters.HTTPAdapter(max_retries=1))

# prepare request information
url = 'http://babelia.libraryofbabel.info/babelia.cgi'
headers = {"Content-Type": "application/x-www-form-urlencoded"}


def download_image_thread(location_q, image_q):
    print("Running Download Image Thread.")
    session = FuturesSession()
    # session.mount("http://", re.adapters.HTTPAdapter(max_retries=1))
    # session.mount("https://", re.adapters.HTTPAdapter(max_retries=1))


    def resolve_request(location):
        start = time.time()
        def resolver(session, r):
            if r.status_code == re.codes.ok:
                image_q.put((location, Image.open(StringIO(r.content))), True)
                stop = time.time()
                print("elapsed time {}".format(stop - start))
            else:
                # TODO: Log this failure
                # Something went wrong with the request
                pass

            location_q.task_done()

        return resolver

    while True:
        print("dl")
        location = location_q.get(True)
        session.post(url, data="location={}".format(location), headers=headers,
                     background_callback=resolve_request(location))

def generate_location_thread(location_q, num_bits):
    print("Running Generate Location Thread.")
    while True:
        print("gen")
        value = random.getrandbits(num_bits)
        location_q.put(value, True)
        if location_q.full():
            print("pausing location gen...")
            location_q.join()
        if location_q.empty():
            print("started location gen...")

        # perhaps unnecesary to save the state
        # state_lock.acquire()
        # state['seed'] = random.getstate()
        # state_lock.release()

def classification_thread(image_q, classifiers, image_path, state, state_lock):
    print("Running Classification Thread")
    iteration = 0
    while True:
        (location, image) = image_q.get(True)
        print("class")

        iteration = iteration + 1
        print("Proccesing image {}".format(iteration))

        regions = detect_interest_regions(image, classifiers)

        print("Found {} regions".format(len(regions)))
        if len(regions) > 0:
            # TODO: stronger unique ids
            unique_id = uuid.uuid4()
            print("Saving image with unique id {}".format(unique_id))
            image.save(os.path.join(image_path, "{}.png".format(unique_id)))

            # Save map of uuid to image location
            state_lock.acquire()
            if state.has_key('interest'):
                temp = state['interest']
                temp[unique_id] = location
                state['interest'] = temp
            else:
                state['interest'] = {unique_id: location}
            state_lock.release()


def spin_crawl_threads(state, classifiers, UPPER_BIT_SIZE, image_path):
    print("Running threads...")
    location_q = Queue(maxsize=512)
    image_q = Queue()
    state_lock = Lock()
    generate_location = Process(target=generate_location_thread,
                                args=(location_q, UPPER_BIT_SIZE))
    classification = Process(target=classification_thread,
                             args=(image_q, classifiers, image_path,
                                   state, state_lock))
    download_image_t = Process(target=download_image_thread,
                             args=(location_q, image_q))

    download_image_t.start()
    classification.start()
    generate_location.start()

    def kill_threads():
        for thread in active_children():
            thread.terminate()

    atexit.register(kill_threads)

    download_image_t.join()

def download_image(value):
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
    parser.add_argument('--max', dest='UPPER_BIT_SIZE', type=int, default=50000,
                        help='The maximum number of bits to generate for an image location')
    parser.add_argument('--threaded', '-t', dest='threaded', action='store_true')
    parser.add_argument('--ping', '-p', dest='ping', action='store_true', default=False)
    parser.set_defaults(threaded=False)

    args = parser.parse_args()
    UPPER_BIT_SIZE = args.UPPER_BIT_SIZE
    shelve_path = args.shelve_path
    image_path = args.image_path
    threaded = args.threaded

    if args.ping:
        while True:
            location = random.getrandbits(UPPER_BIT_SIZE)
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
        spin_crawl_threads(d, classifiers, UPPER_BIT_SIZE, image_path)
    else:
        while True:
            location = random.getrandbits(UPPER_BIT_SIZE)
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


