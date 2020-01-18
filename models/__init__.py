from .rnn_model import RNN_model
from models.unigram import Unigram
from models.cnn_model import CNN
from models.attentive_pooling_cnn import Attentive_CNN

def setup(opt):
    if opt["model_name"] == "rnn":
        model = RNN_model(opt)
    elif opt['model_name'] == 'unigram':
        model = Unigram(opt)
    elif opt['model_name'] == 'cnn':
        if opt['attention'] == 'attentive':
            model = Attentive_CNN(opt)
        else:
            model = CNN(opt)
    else:
        print("no model")
        exit(0)
    return model
