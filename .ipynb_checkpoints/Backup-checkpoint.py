# models.py

##############################################################################
# LIBRARY IMPORTS
##############################################################################
# Torch and related PyTorch libraries
import torch
from torch import nn  # Neural networks
import torch.nn.functional as F  # Functional API
from torch.utils.data import Dataset, DataLoader  # Data utilities

# NLP Preprocessing Libraries (nltk)
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize  # Tokenizers
from nltk.corpus import stopwords  # Stop words
from nltk.stem import PorterStemmer  # Stemmer

# Utility and Helper Libraries
from typing import List  # Type hinting
from collections import Counter  # Counting elements
import re  # Regular expressions
import numpy as np  # Numerical computations
import time  # Time tracking
import argparse  # Argument parsing
import matplotlib.pyplot as plt  # Plotting

##############################################################################
# UTILS
##############################################################################
class Indexer(object):
    """
    Bijection between objects and integers starting at 0. Useful for mapping
    labels, features, etc. into coordinates of a vector space.

    Attributes:
        objs_to_ints
        ints_to_objs
    """
    def __init__(self):
        self.objs_to_ints = {}
        self.ints_to_objs = {}

    def __repr__(self):
        return str([str(self.get_object(i)) for i in range(0, len(self))])

    def __str__(self):
        return self.__repr__()

    def __len__(self):
        return len(self.objs_to_ints)

    def get_object(self, index):
        """
        :param index: integer index to look up
        :return: Returns the object corresponding to the particular index or None if not found
        """
        if (index not in self.ints_to_objs):
            return None
        else:
            return self.ints_to_objs[index]

    def contains(self, object):
        """
        :param object: object to look up
        :return: Returns True if it is in the Indexer, False otherwise
        """
        return self.index_of(object) != -1

    def index_of(self, object):
        """
        :param object: object to look up
        :return: Returns -1 if the object isn't present, index otherwise
        """
        if (object not in self.objs_to_ints):
            return -1
        else:
            return self.objs_to_ints[object]

    def add_and_get_index(self, object, add=True):
        """
        Adds the object to the index if it isn't present, always returns a nonnegative index
        :param object: object to look up or add
        :param add: True by default, False if we shouldn't add the object. If False, equivalent to index_of.
        :return: The index of the object
        """
        if not add:
            return self.index_of(object)
        if (object not in self.objs_to_ints):
            new_idx = len(self.objs_to_ints)
            self.objs_to_ints[object] = new_idx
            self.ints_to_objs[new_idx] = object
        return self.objs_to_ints[object]

##############################################################################
# SENTIMENT DATA WRAPPER CLASSES
##############################################################################
class SentimentExample:
    """
    Data wrapper for a single example for sentiment analysis.

    Attributes:
        words (List[string]): list of words
        label (int): 0 or 1 (0 = negative, 1 = positive)
    """

    def __init__(self, words, label):
        self.words = words
        self.label = label

    def __repr__(self):
        return repr(self.words) + "; label=" + repr(self.label)

    def __str__(self):
        return self.__repr__()


def read_sentiment_examples(infile: str) -> List[SentimentExample]:
    """
    Reads sentiment examples in the format [0 or 1]<TAB>[raw sentence]; tokenizes and cleans the sentences and forms
    SentimentExamples.

    Note that we lowercase the data for you. This is because the GloVe embeddings don't
    distinguish case and so can only be used with lowercasing.

    :param infile: file to read from
    :return: a list of SentimentExamples parsed from the file
    """
    f = open(infile)
    exs = []
    for line in f:
        if len(line.strip()) > 0:
            fields = line.split("\t")
            if len(fields) != 2:
                fields = line.split()
                label = 0 if "0" in fields[0] else 1
                sent = " ".join(fields[1:]).lower()
            else:
                # Slightly more robust to reading bad output than int(fields[0])
                label = 0 if "0" in fields[0] else 1
                sent = fields[1].lower()
            tokenized_cleaned_sent = list(filter(lambda x: x != '', sent.rstrip().split(" ")))
            exs.append(SentimentExample(tokenized_cleaned_sent, label))
    f.close()
    return exs


def read_blind_sst_examples(infile: str) -> List[List[str]]:
    """
    Reads the blind SST test set, which just consists of unlabeled sentences
    :param infile: path to the file to read
    :return: list of tokenized sentences (list of list of strings)
    """
    f = open(infile, encoding='utf-8')
    exs = []
    for line in f:
        if len(line.strip()) > 0:
            exs.append(line.split(" "))
    return exs


def write_sentiment_examples(exs: List[SentimentExample], outfile: str):
    """
    Writes sentiment examples to an output file with one example per line, the predicted label followed by the example.
    Note that what gets written out is tokenized.
    :param exs: the list of SentimentExamples to write
    :param outfile: out path
    :return: None
    """
    o = open(outfile, 'w')
    for ex in exs:
        o.write(repr(ex.label) + "\t" + " ".join([word for word in ex.words]) + "\n")
    o.close()


##############################################################################
# WORD EMBEDDING WRAPPER CLASSES
##############################################################################

class WordEmbeddings:
    """
    Wraps an Indexer and a list of 1-D numpy arrays where each position in the list is the vector for the corresponding
    word in the indexer. The 0 vector is returned if an unknown word is queried.
    """
    def __init__(self, word_indexer, vectors):
        self.word_indexer = word_indexer
        self.vectors = vectors

    def get_initialized_embedding_layer(self, frozen=True):
        """
        :param frozen: True if you want the embedding layer to stay frozen, false to fine-tune embeddings
        :return: torch.nn.Embedding layer you can use in your network
        """
        return torch.nn.Embedding.from_pretrained(torch.FloatTensor(self.vectors), freeze=frozen)

    def get_embedding_length(self):
        return len(self.vectors[0])

    def get_vocab_length(self):
        return len(self.word_indexer)

    def get_embedding(self, word):
        """
        Returns the embedding for a given word
        :param word: The word to look up
        :return: The UNK vector if the word is not in the Indexer or the vector otherwise
        """
        word_idx = self.word_indexer.index_of(word)
        if word_idx != -1:
            return self.vectors[word_idx]
        else:
            return self.vectors[self.word_indexer.index_of("UNK")]


def read_word_embeddings(embeddings_file: str) -> WordEmbeddings:
    """
    Loads the given embeddings (ASCII-formatted) into a WordEmbeddings object. Augments this with an UNK embedding
    that is the 0 vector. Reads in all embeddings with no filtering -- you should only use this for relativized
    word embedding files.
    :param embeddings_file: path to the file containing embeddings
    :return: WordEmbeddings object reflecting the words and their embeddings
    """
    f = open(embeddings_file)
    word_indexer = Indexer()
    vectors = []
    # Make position 0 a PAD token, which can be useful if you
    word_indexer.add_and_get_index("PAD")
    # Make position 1 the UNK token
    word_indexer.add_and_get_index("UNK")
    for line in f:
        if line.strip() != "":
            space_idx = line.find(' ')
            word = line[:space_idx]
            numbers = line[space_idx+1:]
            float_numbers = [float(number_str) for number_str in numbers.split()]
            vector = np.array(float_numbers)
            word_indexer.add_and_get_index(word)
            # Append the PAD and UNK vectors to start. Have to do this weirdly because we need to read the first line
            # of the file to see what the embedding dim is
            if len(vectors) == 0:
                vectors.append(np.zeros(vector.shape[0]))
                vectors.append(np.zeros(vector.shape[0]))
            vectors.append(vector)
    f.close()
    print("Read in " + repr(len(word_indexer)) + " vectors of size " + repr(vectors[0].shape[0]))
    # Turn vectors into a 2-D numpy array
    return WordEmbeddings(word_indexer, np.array(vectors))

def get_glove_embeddings(dim):
    """
    Retrieve glove vector representations based on specific vector dimensions (50 or 300).
    :param dim: specifies dimensionality of the vector embeddings
    :return: WordEmbeddings object reflecting the words and their glove embeddings
    """
    if dim == 300:
        # Get 300 Dim
        glove_file = './data/glove.6B.300d-relativized.txt'
    else:
        # get 50 dim
        glove_file = './data/glove.6B.50d-relativized.txt'
    return read_word_embeddings(glove_file)

#
def glovedoc_to_avg_vec(doc, glove):
    """
    This function produces a glove vector representation for the entire text document by averaging
    individual word vectors
    :param doc: A sentence or text represented as a tokenized list of words
    :param glove: WordEmbeddings object reflecting the words and their glove embeddings
    :return: A vector (NumPy Array) produced by averaging individual word vectors in a text document. 
    """
    vecs = []
    for token in doc:
        try:
            vecs.append(glove.get_embedding(token))
        except KeyError:
            pass
    return np.mean(vecs, axis=0)

##############################################################################
#PRE PROCESSING AND FEATURE REPRESENTATION
##############################################################################

#Doing some preprocessing on the data
ps = PorterStemmer() #Stemmer
stop_words = set(stopwords.words('english')) #Stop words list
nltk.download('punkt_tab')

# return a list of tokens
def pre_processing_by_nltk(words):
    """
    This function performs additional preprocessing such as stemming, lower-casing, and stop-word removal 
    on a tokenized list of words
    :param words: A sentence or text represented as a tokenized list of words
    :return: A tokenized and cleaned list of lower cased words
    """
    #stemming, stop word removal, and converting to lower case
    return [(ps.stem(word)).lower() for word in words if word.lower() not in stop_words]


class SentimentDatasetDAN(Dataset):
    def __init__(self, infile, vectorizer=None, glovedim=50):
        """
        Initialize the dataset by reading examples and converting them to embeddings
        :param infile: Path to the data file containing text and labels
        :param word_embeddings: A WordEmbeddings object for converting words to indices
        :param glovedim: Dimensionality of GloVe embeddings
        """
        # Read the sentiment examples (you need to implement read_sentiment_examples for your dataset format)
        self.examples = read_sentiment_examples(infile)  # Assumes each example has words and a label

        # Preprocess and tokenize the text examples (optional, you can customize pre_processing_by_nltk)
        self.docs = [pre_processing_by_nltk(ex.words) for ex in self.examples]
        self.labels = [ex.label for ex in self.examples]
        
        # Vectorize the sentences using CountVectorizer
        if vectorizer == 'glove':
            glove = get_glove_embeddings(glovedim)
            self.embeddings = np.array([glovedoc_to_avg_vec(doc, glove) for doc in self.docs])
        else:
            print('TODO: Random embeddings')
        
        # Convert embeddings and labels to PyTorch tensors
        self.embeddings = torch.tensor(self.embeddings, dtype=torch.float32)
        self.labels = torch.tensor(self.labels, dtype=torch.long)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        # Return the feature vector and label for the given index
        return self.embeddings[idx], self.labels[idx]



##############################################################################
# NEURAL NETWORK ARCHITECTURES
##############################################################################

# Two-layer fully connected neural network
class NN2DAN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.dropout1 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(hidden_size, 2)
        

    def forward(self, x):
        x = self.dropout1(x)
        x = F.tanh(self.fc1(x))
        x = self.dropout2(x)
        x = F.sigmoid(self.fc2(x))
        
        return x

    
# Three-layer fully connected neural network
class NN3DAN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, 2)
        # self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = F.tanh(self.fc1(x))
        x = self.dropout1(x)
        x = F.tanh(self.fc2(x))
        x = self.dropout2(x)
        x = F.relu(self.fc3(x))
        x = F.sigmoid(self.fc4(x))
        return x

##############################################################################
# TRAINING, EVALUATION AND FINE-TUNING
##############################################################################

# Training function
def train_epoch(data_loader, model, loss_fn, optimizer):
    size = len(data_loader.dataset)
    num_batches = len(data_loader)
    model.train()
    train_loss, correct = 0, 0
    for batch, (X, y) in enumerate(data_loader):
        X = X.float()

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
        train_loss += loss.item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    average_train_loss = train_loss / num_batches
    accuracy = correct / size
    return accuracy, average_train_loss


# Evaluation function
def eval_epoch(data_loader, model, loss_fn, optimizer):
    size = len(data_loader.dataset)
    num_batches = len(data_loader)
    model.eval()
    eval_loss = 0
    correct = 0
    for batch, (X, y) in enumerate(data_loader):
        X = X.float()

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
        eval_loss += loss.item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    average_eval_loss = eval_loss / num_batches
    accuracy = correct / size
    return accuracy, average_eval_loss


# Experiment function to run training and evaluation for multiple epochs
def experiment(model, train_loader, test_loader):
    # Possible variations
    # loss_functions = ['']
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.05)
    #optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)


    all_train_accuracy = []
    all_test_accuracy = []
    for epoch in range(100):
        train_accuracy, train_loss = train_epoch(train_loader, model, loss_fn, optimizer)
        all_train_accuracy.append(train_accuracy)

        test_accuracy, test_loss = eval_epoch(test_loader, model, loss_fn, optimizer)
        all_test_accuracy.append(test_accuracy)

        if epoch % 10 == 9:
            print(f'Epoch #{epoch + 1}: train accuracy {train_accuracy:.3f}, dev accuracy {test_accuracy:.3f}')
    
    return all_train_accuracy, all_test_accuracy

def plot_accuracy(*args, **kwargs):
    # Create a Figure
    filename=None
    plt.figure(figsize=(8, 6))

    # Plot lines
    for arg in args:
        plt.plot(arg['accuracy'], label=arg['label'])

    plt.xlabel('Epochs')
    
    for key, value in kwargs.items():
        if key == 'type' and value == 'train':
            plt.ylabel('Training Accuracy')
        elif key == 'type' and value == 'test':
            plt.ylabel('Testing Accuracy')
        elif key == 'title':
            plt.title(value)
        elif key == 'file':
            filename = value
            
    plt.legend()
    plt.grid()

    plt.savefig(filename)
    print(f"\nPlot saved as {filename}")
    

def main():

    # Set up argument parser
    # parser = argparse.ArgumentParser(description='Run model training based on specified model type')
    # parser.add_argument('--model', type=str, required=True, help='Model type to train (e.g., BOW)')

    # Parse the command-line arguments
    # args = parser.parse_args()

    # Load dataset
    glovedim=300
    
    start_time = time.time()

    train_data = SentimentDatasetDAN("data/train.txt", glovedim=glovedim, 'glove')
    dev_data = SentimentDatasetDAN("data/dev.txt", glovedim=glovedim, 'glove')
    train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
    test_loader = DataLoader(dev_data, batch_size=16, shuffle=False)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Data loaded in : {elapsed_time} seconds")

    # Train and evaluate NN2
    start_time = time.time()
    print('\n2 layers:')
    nn2_train_accuracy, nn2_test_accuracy = experiment(NN2DAN(input_size=glovedim, hidden_size=100), train_loader, test_loader)

    #Train and evaluate NN3
    print('\n3 layers:')
    nn3_train_accuracy, nn3_test_accuracy = experiment(NN3DAN(input_size=glovedim, hidden_size=100), train_loader, test_loader)

    plot_accuracy({'accuracy': nn2_train_accuracy, 'label': '2 layers'}, 
                 {'accuracy': nn3_train_accuracy, 'label': '3 layers'},
                 type='train', title='Training Accuracy for 2, 3 Layer Networks', file='train_accuracy.png')
    
    plot_accuracy({'accuracy': nn2_test_accuracy, 'label': '2 layers'}, 
                 {'accuracy': nn3_test_accuracy, 'label': '3 layers'},
                 type='test', title='Test Accuracy for 2, 3 Layer Networks', file='test_accuracy.png')

if __name__ == "__main__":
    main()