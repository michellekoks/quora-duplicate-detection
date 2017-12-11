STEPS TO RUN THE MODEL:

# Download the data files (train.csv, test.csv, train_label.csv) and those 2 files, put into the "./data" folder:
  1.ready-made spelling correction by YuriyGuts
  ## https://github.com/YuriyGuts/kaggle-quora-question-pairs/blob/master/data/aux/spelling_corrections.json
  2.pre-train word vector from GoogleNews
  ## https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing

# Run the 1.Preprocess.py
  You should expect the following files to be generated:
	"./data/token_train.pkl"
	"./data/token_test.pkl"

# Run the 2.Google_embedding preparation.py 
  You should expect the following files to be generated:
	"/data/embeddings.npy" - embedding matrix
	"/data/vocabulary.npy" - the dictionary to lookup word --> embedding matrix index
	"/data/inverse_vocabulary.npy" - the inverse dictionary to lookup word --> embedding matrix index
	"/data/token_train_n.pkl" - the token vector of the embedding index for each word for train data
	"/data/token_test_n.pkl" - the token vector of the embedding index for each word for test data

# Run the 3.Final_model.py
  The model will train first on the test and validation data (from the whole train dataset that we are provided) and
  train the model again with the whole train data to generate the submission as '/submission.csv'
	

# Other files:
 "util.py" : functions for pre-processing and other pre-defined functions