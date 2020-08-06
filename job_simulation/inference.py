import re
from collections import OrderedDict
import fasttext
import cnf
import numpy as np
from preprocessing import preprocess_text, preprocess_text_advanced

def inference_svm(text, vectorizer, clf):
    cleaned_text = preprocess_text(text)
    cleaned_text = preprocess_text_advanced(cleaned_text, stem = True)
    embed = vectorizer.transform([cleaned_text])
    result = clf.predict(embed)[0]
    return {cnf.map_dct[str(result)]: ''}

def inference_fasttext(text, model):
    result = {}
    cleaned_text = preprocess_text(text)
    cleaned_text = preprocess_text_advanced(cleaned_text, stem=True)
    labels, probs = model.predict(cleaned_text, k = 7)
    lbls = [int(i) for i in re.findall(r'\d+', ' '.join(list(labels)))]
    i = 0
    for l in lbls:
        result[cnf.map_dct[str(l)]] = str(round(probs[i], 3))
        i += 1
    return result