import json
import os
os.remove("parsed.csv")
f = open("reviews_Video_Games_5.json");

data = [];
def reviewCreator(overall, text, summary):
    x = {
        overall: overall,
        text: text,
        summary: summary
    }
    return x
    #print parts[4]
g = open("parsed.csv", "a")

for i in range(100000):
    dict = json.loads(f.readline())
    data.append(reviewCreator(dict["overall"],dict["reviewText"],dict["summary"]))
    g.write(str(dict["overall"]) + " | " + dict["reviewText"] + " | " + dict["summary"] + "\n \n")
#print data
