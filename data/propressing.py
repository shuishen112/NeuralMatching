import numpy as np 
import spacy 
from spacy.lang.en import English
nlp = spacy.load('en_core_web_md')
data_file = "wiki/train.txt"
import sys
sys.path.append('../')
with open(data_file,'r') as fin:
    with open('wiki/new_train.txt','w') as fout:
        for e,line in enumerate(fin):
            if e > 10:
                break
            fout.write(str(nlp(line)))