from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
import numpy as np
from scipy.special import softmax
import csv
import urllib.request


def detect_mood_roberta(text, text_model):
    task = 'sentiment'
    if text_model=='twitter-roberta':
        MODEL = f"cardiffnlp/twitter-roberta-base-{task}"
    else:
        MODEL = f"finiteautomata/bertweet-base-{task}-analysis" 
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    mapping_link = f"https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/{task}/mapping.txt"
    with urllib.request.urlopen(mapping_link) as f:
        html = f.read().decode('utf-8').split("\n")
        csvreader = csv.reader(html, delimiter='\t')
    labels = [row[1] for row in csvreader if len(row) > 1]

    model = AutoModelForSequenceClassification.from_pretrained(MODEL)
    model.save_pretrained(MODEL)
    tokenizer.save_pretrained(MODEL)

    encoded_input = tokenizer(text, return_tensors='pt')
    output = model(**encoded_input)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)

    ranking = np.argsort(scores)
    ranking = ranking[::-1]
    l = labels[ranking[0]]
    s = np.round(float(scores[ranking[0]]), 4)
    if l=='neutral':
        return l,s, 'static/neutral.png'
    elif l=='positive':
        return l,s, 'static/positive.png'
    return l,s, 'static/negative.jpg' 
    