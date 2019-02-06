import cv2
import numpy as np
import os.path
import os
import pickle
import argparse
import face_recognition


# Load key embeddings

database = pickle.load( open( "database.p", "rb" ))

# Parse arguments

parser = argparse.ArgumentParser()
parser.add_argument("--path", help='point to path of picture')
args = parser.parse_args()

path = args.path
print('Path: '+args.path)

# Detect face & generate embedding

def generate(path):
    try:

        image = cv2.imread(path)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        boxes = face_recognition.face_locations(rgb, model='hog')
        encodings = face_recognition.face_encodings(rgb, boxes)
        print('number of faces detected: ' + str(len(encodings)))
        return encodings

    except Exception as e:
        #print(e)
        pass
    # cv2.destroyAllWindows()

# Verify who it is

def verify(path, database):
    embeddings = generate(path)

    for emb, num in zip(embeddings, range(len(embeddings))):

        min_dist = 0.60

        for (name, db_emb) in database.items():
            dist = np.linalg.norm(emb - db_emb)

            if dist < min_dist:
                min_dist = dist
                identity = name

        if min_dist >= 0.60:
            print("Person " + str(num) + " is not in the database.")
        else:
            print("Person " + str(num) + " is " + str(identity) + ", the distance is " + str(min_dist))

verify(path, database)

