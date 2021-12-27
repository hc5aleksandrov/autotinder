import requests
import datetime
from geopy.geocoders import Nominatim
#from time import sleep
#from random import random
from likeliness_classifier import Classifier
import person_detector
import tensorflow as tf
#import time
import time 

import random
from tinder_py.tinder.tinder import TinderClient


TINDER_URL = "https://api.gotinder.com"
geolocator = Nominatim(user_agent="auto-tinder")
PROF_FILE = "./images/unclassified/profiles.txt"

class tinderAPI():

    def __init__(self, token):
        self._token = token

    def profile(self):
        data = requests.get(TINDER_URL + "/v2/profile?include=account%2Cuser", headers={"X-Auth-Token": self._token}).json()
        return Profile(data["data"], self)

    def matches(self, limit=10):
        data = requests.get(TINDER_URL + f"/v2/matches?count={limit}", headers={"X-Auth-Token": self._token}).json()
        return list(map(lambda match: Person(match["person"], self), data["data"]["matches"]))

    def like(self, user_id):
        data = requests.get(TINDER_URL + f"/like/{user_id}", headers={"X-Auth-Token": self._token}).json()
        return {
            "is_match": data["match"],
            "liked_remaining": data["likes_remaining"]
        }

    def dislike(self, user_id):
        requests.get(TINDER_URL + f"/pass/{user_id}", headers={"X-Auth-Token": self._token}).json()
        return True

    def nearby_persons(self):
        print('get: ', TINDER_URL + "/v2/recs/core", self._token)
        data = requests.get(TINDER_URL + "/v2/recs/core", headers={"X-Auth-Token": self._token}).json()
        print('data: ', data)
        return list(map(lambda user: Person(user["user"], self), data["data"]["results"]))


def predict(person, classifier, sess):
        ratings = []
        i = 0
        for image in person.photos:
            i += 1
            req = requests.get(image.url, stream=True)
            tmp_filename = f"./images/tmp/run.jpg"     
            imgname = person.id + "_" + str(i) + '.jpg'
            save_img_path = f"./images/unclassified/%s" % imgname
            if req.status_code == 200:
                with open(tmp_filename, "wb") as f:
                    f.write(req.content)
                    
                with open(save_img_path, "wb") as f2:
                    f2.write(req.content)
                   
            img = person_detector.get_person(tmp_filename, sess)
            if img:
                img = img.convert('L')
                img.save(tmp_filename, "jpeg")
                certainty = classifier.classify(tmp_filename)
                pos = certainty["positive"]
                ratings.append(pos)
        ratings.sort(reverse=True)
        ratings = ratings[:5]
        if len(ratings) == 0:
            return 0.001
        if len(ratings) == 1:
            return ratings[0]
        return ratings[0]*0.6 + sum(ratings[1:])/len(ratings[1:])*0.4


if __name__ == "__main__":
    token = "0e8d75b7-48cb-4ba5-8baf-9e1aa1dfe806"#"87ffb204-dad2-41c0-a039-62de457616d2"
    #api = tinderAPI(token)
    api = TinderClient(token)
    print('API: ', api)
    detection_graph = person_detector.open_graph()
    with detection_graph.as_default():
        with tf.compat.v1.Session() as sess:

            classifier = Classifier(graph="./tf/training_output/retrained_graph.pb",
                                    labels="./tf/training_output/retrained_labels.txt")

            end_time = 1640213115 + 60*60*2.8
            while True:#time() < end_time:
                
                    print(f"------ TIME LEFT: {(end_time - time.time())/60} min -----")
                    persons = api.get_recommendations()
                    pos_schools = ["Universität Zürich", "University of Zurich", "UZH", "HWZ Hochschule für Wirtschaft Zürich",
                                   "ETH Zürich", "ETH Zurich", "ETH", "ETHZ", "Hochschule Luzern", "HSLU", "ZHAW",
                                   "Zürcher Hochschule für Angewandte Wissenschaften", "Universität Bern", "Uni Bern",
                                   "PHLU", "PH Luzern", "Fachhochschule Luzern", "Eidgenössische Technische Hochschule Zürich"]
                    print('len PERSONS: ', len(persons)) 
                    print("persons:",persons)
                    try:
                        for person in persons:
                            print("___person:",person)
                            score = predict(person, classifier, sess)
                            #print('score: ', score) 
                            #for school in pos_schools:
                            #    if school in person.schools:
                            #        print()
                            #        score *= 1.2

                            print("-------------------------")
                            print("ID: ", person.id)
                            print("Name: ", person.name)
                            #print("Schools: ", person.schools)
                            #print("Images: ", person.photos)
                            print(score)

                            if score > 0.6:
                                res = person.like()
                                print("LIKE")
                                print("Response: ", res)
                            else:
                                res = person.dislike()
                                print("DISLIKE")
                                print("Response: ", res)
                    except Exception:
                        pass
                    time.sleep(random.randint(3, 10) )                        
                




    classifier.close()
