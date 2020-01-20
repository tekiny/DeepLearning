


import urllib
import urllib.request
import cv2
import os
import numpy as np
from multiprocessing.dummy import Pool as ThreadPool
import itertools
import pdb

pic_num = 1


def store_raw_images(paths, links):
    global pic_num
    for link, path in zip(links, paths):
        if not os.path.exists(path):
            os.makedirs(path)
        image_urls = str(urllib.request.urlopen(link).read())

        pool = ThreadPool(32)
        pool.starmap(loadImage, zip(itertools.repeat(path), image_urls.split('\\n'), itertools.count(pic_num)))
        pool.close()
        pool.join()


def loadImage(path, link, counter):
    global pic_num
    if pic_num < counter:
        pic_num = counter + 1;
    try:
        urllib.request.urlretrieve(link, path + "/" + str(counter) + ".jpg")
        img = cv2.imread(path + "/" + str(counter) + ".jpg")
        if img is not None:
            cv2.imwrite(path + "/" + str(counter) + ".jpg", img)
            print(counter)

    except Exception as e:
        print(str(e))


def removeInvalid(dirPaths, invalid_path):
    for dirPath in dirPaths:
        for img in os.listdir(dirPath):
            for invalid in os.listdir(invalid_path):
                try:
                    current_image_path = str(dirPath) + '/' + str(img)
                    invalid = cv2.imread(invalid_path + '/' + str(invalid))
                    question = cv2.imread(current_image_path)
                    if (question is None) or (invalid.shape == question.shape and not (np.bitwise_xor(invalid, question).any())):
                        os.remove(current_image_path)
                        break

                except Exception as e:
                    print(str(e))


links = [
        'http://image-net.org/api/text/imagenet.synset.geturls?wnid=n01318894',
        'http://image-net.org/api/text/imagenet.synset.geturls?wnid=n03405725',
        'http://image-net.org/api/text/imagenet.synset.geturls?wnid=n07942152',
        'http://image-net.org/api/text/imagenet.synset.geturls?wnid=n00021265',
        'http://image-net.org/api/text/imagenet.synset.geturls?wnid=n07690019',
        'http://image-net.org/api/text/imagenet.synset.geturls?wnid=n07865105',
        'http://image-net.org/api/text/imagenet.synset.geturls?wnid=n07697537']

paths = ['/Users/gryslik/temp/ProcessedPictures/not_bitcoin/pets/',
         '/Users/gryslik/temp/ProcessedPictures/not_bitcoin/furniture/',
         '/Users/gryslik/temp/ProcessedPictures/not_bitcoin/people/',
         '/Users/gryslik/temp/ProcessedPictures/not_bitcoin/food/',
         '/Users/gryslik/temp/ProcessedPictures/bitcoin/frankfurter/',
         '/Users/gryslik/temp/ProcessedPictures/bitcoin/chili-dog/',
         '/Users/gryslik/temp/ProcessedPictures/bitcoin/bitcoin/']

store_raw_images(paths, links)
removeInvalid(paths, '/Users/gryslik/temp/invalid/')
