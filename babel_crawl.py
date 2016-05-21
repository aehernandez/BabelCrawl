import requests as re
import numpy as np
import cv2
import random
from PIL import Image
from StringIO import StringIO

import uuid
import os
import shelve
import atexit
import argparse

def download_image(value):
    url = 'http://babelia.libraryofbabel.info/babelia.cgi'
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    r = re.post(url, data="location={}".format(value), headers=headers)
    return Image.open(StringIO(r.content))

cascade_path = 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(cascade_path)

def load_classifiers(paths):
    classifiers = []
    for path in paths:
        print("loading classifier {}".format(path))
        classifiers.append(cv2.CascadeClassifier(path))
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
            flags = cv2.cv.CV_HAAR_SCALE_IMAGE
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


# detect_face(Image.open('woman.jpeg'), classifiers)
# detect_face(download_image(136136713710378107810781), classifiers)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Crawl http://babelia.libraryofbabel.info/ for faces and other interesting features')
    parser.add_argument('--state_path', dest='shelve_path', default='state.crawl', type=str,
                        help='The path to save the persistent state dictionary used by this application')
    parser.add_argument('--image_path', dest='image_path', default='./images', type=str,
                        help='The path to save images which contain interesting regions')
    parser.add_argument('--max', dest='UPPER_BIT_SIZE', type=int, default=50000,
                        help='The maximum number of bits to generate for an image location')

    args = parser.parse_args()
    UPPER_BIT_SIZE = args.UPPER_BIT_SIZE
    shelve_path = args.shelve_path
    image_path = args.image_path

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

    classifier_paths = map(lambda s: './classifiers/haarcascade_' + s,
                           classifier_paths)
    classifiers = load_classifiers(classifier_paths)

    while True:
        location = random.getrandbits(UPPER_BIT_SIZE)
        iteration = iteration + 1
        print("Processed {} images from seed".format(iteration))

        image = download_image(location)
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
