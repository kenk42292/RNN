from utils import *
from RNN import *

from config import *


if preprocess_data:
    X_train, Y_train, index_to_word, word_to_index \
        = preprocess(vocabulary_size, training_data_raw)
    with open(training_data_preprocessed, "w") as f:
        pickle.dump([X_train, Y_train, index_to_word, word_to_index], f)
else:
    with open(training_data_preprocessed, "rU") as f:
        X_train, Y_train, index_to_word, word_to_index = pickle.load(f)
    
model = RNN(vocabulary_size, hidden_dim)

if (training_choice==REUSE_PARAMS) or (training_choice==CONTINUE_TRAIN): 
    with open(model_file, "rU") as f:
        model = pickle.load(f)
if (training_choice==CONTINUE_TRAIN) or (training_choice==RESET_PARAMS):    
    model.train(X_train[:1000], Y_train[:1000], nepoch, learning_rate, batch_size)
    with open(model_file, "w") as f:
        pickle.dump(model, f)





