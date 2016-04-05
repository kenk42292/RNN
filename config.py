
### Data storage files
training_data_raw = "data/reddit_text.csv"
training_data_preprocessed = "training_data.pickle"
model_file = "model.pickle"

### Stored data reuse
preprocess_data = False

RESET_PARAMS = 0
CONTINUE_TRAIN = 1
REUSE_PARAMS = 2
training_choice = RESET_PARAMS 

### Model Parameters
vocabulary_size = 8000
hidden_dim = 100

### Learning Paramters
nepoch = 10000
learning_rate = 0.01
batch_size = 5




