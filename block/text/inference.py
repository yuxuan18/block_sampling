from transformers import pipeline
from typing import List
import spacy
from tqdm import tqdm

def sentiment_analysis(sentences: List[str], 
                       batch_size: int=128, model: str="cardiffnlp/twitter-roberta-base-sentiment"):
    print("Executing sentiment analysis ...")
    sentiment_pipeline = pipeline("sentiment-analysis", model=model, device_map="auto")
    results = []
    for result in tqdm(sentiment_pipeline(sentences, batch_size=batch_size)):
        results.append(result)
    return results


def entity_recognition(sentences: List[str], 
                        batch_size: int=128, model: str="dslim/bert-base-NER"):
    if spacy.prefer_gpu():
        print(f"Executing entity recognition on {len(sentences)} samples with GPU enabled ...")
    else:
        print(f"Executing entity recognition on {len(sentences)} samples with GPU disabled ...")
    nlp = spacy.load("en_core_web_sm")
    results = []
    for result in nlp.pipe(sentences, batch_size=batch_size):
        results.append([(ent.text, ent.label_) for ent in result.ents])
    return results

if __name__ == "__main__":
    import pandas as pd
    twitter = pd.read_csv("../../data/twitter.csv", low_memory=False)
    twitter["Tweet"] = twitter["Tweet"].apply(lambda x: str(x) if len(str(x)) > 0 else "None")
    tweets = twitter["Tweet"].to_list()
    print(entity_recognition(tweets[:1000]))