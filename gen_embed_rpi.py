import cv2
import os.path
import pickle
import argparse
import imutils
import os
import face_recognition

## Detect faces and generate embedding

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

# Create database

database = {}

# Generate and store embeddings from individual photo folders

def retrieve_photos(path):
    paths = []
    for i in os.listdir(path):
        path_full = os.path.join(path, i)
        paths.append(path_full)
    return paths

def generate_from_folder(directory, name):
    embeddings = []
    photos = retrieve_photos(directory)
    for i in photos:
        try:
            embedding = generate(str(i))
            embeddings.append(embedding[0])
        except:
            pass

    for i in range(len(embeddings)):
        database[str(name) + '_' + str(i)] = embeddings[i]
        print('appended embedding '+name+'_'+str(i))

    print('embeddings appended to database for '+name)

# Example for single person photos

generate_from_folder('pictures/Mike', 'Mike')

# Generate and store 'team photo' embeddings

path_team = 'pictures/team/team_photo.png'
embeddings_team = generate(path_team)

database["Mike"] = embeddings_team[0]
database["Lieke"] = embeddings_team[1]
database["Richard"] = embeddings_team[2]

# Pickle key embeddings / save

pickle.dump(database, open("database.p", "wb"))
