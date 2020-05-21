import json



with open('data.json', 'r', encoding='UTF-8') as json_file:
    json_string = json.load(json_file)

def body_load():
    for i in json_string:
       yield i

