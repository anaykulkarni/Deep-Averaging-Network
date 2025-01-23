#***********************************************************************************************************************
# models.py                                                                                                            |
# Author: Anay Kulkarni (ankulkarni@ucsd.edu)                                                                          |
# Date: October 15, 2024                                                                                               |
########################################################################################################################        
#                                            Assignment 1 Code                                                         |
########################################################################################################################
#                                       ----------- README -----------                                                 |
#                                                                                                                      |
# This file is designed to work as a standalone program. All supplementary functions provided in the PA1 zip folder    |
# have been implemented in this file.                                                                                  |
#                                                                                                                      |
#                                                                                                                      |
# The code for each subquestion in all parts of the assignment can be accessed through arguments passed to python      | 
# shell. The arguments follow the order of the questions i.e., 1a, 1b, 2a                                              |           
#                                                                                                                      |
# python DANmodels.py --model 1a      #Runs the code for fine-tuning DANs with Glove Embeddings (4 experiments)        |
# python DANmodels.py --model 1b      #Runs the code for Glove vs Random weights                                       |
# python DANmodels.py --model 2a      #Runs the code for BPE encoding (Subword vs Word) and (performance vs vocab size)|
# python DANmodels.py --model all     #Runs all experiments back to back (WARNING: the code will                       |
#                                     run for a long time)                                                             |
#                                                                                                                      | 
#                                                                                                                      |
# By Default the code uses 100 epochs for each experiment. You can change this by updating the for loop range in the   |
# experiment() function. This will allow the code to run faster at potential cost to performance of the model.         |      #                                                                                                                      |      
#                                                                                                                      |
# References:                                                                                                          |
# BPE code uses algorithm referenced in the lecture slide                                                              |
#                                                                                                                      |
#***********************************************************************************************************************



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
import collections
from collections import Counter, defaultdict  # Counting elements
import re  # Regular expressions
import numpy as np  # Numerical computations
import time  # Time tracking
import argparse  # Argument parsing
import matplotlib.pyplot as plt  # Plotting
from torch.nn.utils.rnn import pad_sequence
import string


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

def collate_fn(batch):
    """
    This function is called by the data loader to pad batches of data. Length of each doc will be 
    equal to the length of the longest document
    :param batch: a batch of input containing tensors of uneven lengths
    :return: a tensor of even length padded with 0s
    """
    # Split batch into sequences and labels
    sequences, labels = zip(*batch)
    
    # Pad sequences to the same length
    padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=0)
    
    # Convert labels to a tensor
    labels = torch.tensor(labels)
    
    return padded_sequences, labels
    
class SentimentDatasetDAN(Dataset):
    def __init__(self, infile, vectorizer=None, glovedim=300):
        """
        Initialize the dataset by reading examples and converting them to word indices as input to NN
        :param infile: Path to the data file containing text and labels
        :param vectorizer: A str sprecifying which vectorizer to use (e.g. GloVe, random)
        :param glovedim: Dimensionality of GloVe embeddings
        """
        # Read the sentiment examples (you need to implement read_sentiment_examples for your dataset format)
        self.examples = read_sentiment_examples(infile)  # Assumes each example has words and a label

        # Preprocess and tokenize the text examples (optional, you can customize pre_processing_by_nltk)
        # self.docs = [pre_processing_by_nltk(ex.words) for ex in self.examples]
        self.docs = [ex.words for ex in self.examples]
        self.labels = [ex.label for ex in self.examples]
        
        # Vectorize the sentences using CountVectorizer
        if vectorizer == 'glove':
            #Get glove embeddings from file based on dimensions specified
            glove = get_glove_embeddings(glovedim)
            # Convert words to indices based on GloVe's word_indexer
            self.word_indexer = glove.word_indexer
            self.embeddings_matrix = glove.vectors
            self.word_indices = [[self.word_indexer.index_of(word) if self.word_indexer.index_of(word) != -1 
                                  else self.word_indexer.index_of("UNK") for word in doc] for doc in self.docs]
        else:
            glove = get_glove_embeddings(glovedim)
            self.word_indexer = glove.word_indexer
            # Initialize the embedding layer with random weights
            self.embeddings_matrix = nn.Embedding(len(glove.word_indexer), glove.vectors[0].shape[0]) #shape(14923, 300)
            self.word_indices = [[self.word_indexer.index_of(word) if self.word_indexer.index_of(word) != -1 
                                  else self.word_indexer.index_of("UNK") for word in doc] for doc in self.docs]
        
        # Convert labels to PyTorch tensors
        self.labels = torch.tensor(self.labels, dtype=torch.long)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        # Get the indices of words for a specific example
        # This tensor is tensor of unevenly long indices. Padding is needed to make input of fixed size
        word_indices = torch.tensor(self.word_indices[idx], dtype=torch.long)
        # Get the corresponding label
        label = self.labels[idx]
        return word_indices, label

class SentimentDatasetDANWithBPE(Dataset):
    def __init__(self, infile, subword_vocab, vectordim=300):
        """
        Initialize the dataset by reading examples, building BPE Vocab, Tokenizing, 
        and converting them to word indices as input to NNs
        :param infile: Path to the data file containing text and labels
        :param subword_vocab: the subword_vocab generated on the training set
        :param vectordim: Dimensionality of dense vector embeddings
        """
        # Read the sentiment examples (you need to implement read_sentiment_examples for your dataset format)
        self.examples = read_sentiment_examples(infile)  # Assumes each example has words and a label
        # Tokenize the text using subword vocabulary
        self.docs = [tokenize_bpe(' '.join(ex.words), subword_vocab).split() for ex in self.examples]
        # Get labels
        self.labels = [ex.label for ex in self.examples]
        # Convert words to indices based on subword vocabulary index
        vocab_list = list(subword_vocab.keys())
        self.word_indices = []
        for doc in self.docs:
            temp = []
            for word in doc:
                try:
                    temp.append(vocab_list.index(word))
                except ValueError:
                    temp.append(vocab_list.index("UNK"))
            self.word_indices.append(temp)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        # Get the indices of words for a specific example
        # This tensor is tensor of unevenly long indices. Padding is needed to make input of fixed size
        word_indices = torch.tensor(self.word_indices[idx], dtype=torch.long)
        # Get the corresponding label
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return word_indices, label

##############################################################################
# BPE IMPLEMENTATION
##############################################################################

# Function to get symbol pairs and their frequencies
def get_stats(vocab):
    pairs = collections.defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            pairs[symbols[i], symbols[i + 1]] += freq
    return pairs

# Function to merge the most frequent pair
def merge_vocab(pair, v_in):
    v_out = {}
    bigram = re.escape(' '.join(pair))
    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
    for word in v_in:
        w_out = p.sub(''.join(pair), word)
        v_out[w_out] = v_in[word]
    return v_out

# Number of merge operations
# num_merges = 10

def initialize_vocab(infile):
    vocab = defaultdict(int)
    exs = read_sentiment_examples(infile)
    for ex in exs:
        for word in ex.words:
            subword_format = ' '.join(list(word)) + ' </w>'
            vocab[subword_format] += 1
    return vocab

def perform_bpe_merges(num_merges, vocab): 
    for i in range(num_merges):
        pairs = get_stats(vocab)
        best = max(pairs, key=pairs.get)
        vocab = merge_vocab(best, vocab)
        #print(f'Merge {i + 1}: {best}')
    return vocab

def get_subword_vocab(bpe_vocab):
    subword_vocab = defaultdict(int)
    #Add the UNK token to the subword vocabulary
    subword_vocab["UNK"] = 0
    for word, count in bpe_vocab.items():
        # Remove the '</w>' marker
        subwords = word.replace('</w>', '').split()
        # Count each subword
        for subword in subwords:
            subword_vocab[subword] += count
    return subword_vocab

def build_bpe_subword_vocab(infile, merges):
    word_vocab = initialize_vocab(infile)
    pre_bpe_subword_vocab = get_subword_vocab(word_vocab)

    print('Before running BPE algorithm-------')
    print('\tWord-level vocab size: ', len(word_vocab))
    print('\tSubword-level vocab size: ', len(pre_bpe_subword_vocab))

    word_vocab = perform_bpe_merges(merges, word_vocab)
    post_bpe_subword_vocab = get_subword_vocab(word_vocab)

    print('After running BPE algorithm--------')
    print('\tWord-level vocab size: ', len(word_vocab))
    print('\tSubword-level vocab size: ', len(post_bpe_subword_vocab))
    print('\nCompression Ratio: ', len(word_vocab)/len(post_bpe_subword_vocab))

    return post_bpe_subword_vocab

def tokenize_bpe(text, vocab):
    # Tokenize text using BPE vocabulary
    words = text.split()
    tokenized_text = []

    # For each word in the list of words
    for word in words:
        subword = []
        i = 0
        # i index starts from beginning of word 
        while i < len(word):
            # Look for the longest subword in the vocabulary that matches the word prefix
            found = False
            #  and j index from the back.
            for j in range(len(word), i, -1):
                sub_token = word[i:j]
                if sub_token in vocab:
                    subword.append(sub_token)
                    i = j  # Move index past the subword
                    found = True
                    break
            if not found:
                subword.append(word[i])  # Add the character if no subword is found
                i += 1
                
        tokenized_text.append(' '.join(subword)) # Join as a sentence of subwords
    
    return ' '.join(tokenized_text) # return tokenized text as a sentence(s) of subwords
        

##############################################################################
# NEURAL NETWORK ARCHITECTURES
##############################################################################
# PART 1
##############################################################################


# A DAN with 2 (trainable) fully-connected layers, and embedding, dropout, and softmax layers
# Uses negative log likelihood loss function with ReLu activation functions
class NNDAN1(nn.Module):
    def __init__(self, embedding_matrix, hidden_size, fine_tune):
        super().__init__()
        # Embedding layer using pre-trained GloVe embeddings
        # If fine_tune is False, the embedding weights are frozen
        self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(embedding_matrix), freeze=not fine_tune)
        # First fully connected layer: Transforms from embedding size to hidden size
        self.fc1 = nn.Linear(embedding_matrix.shape[1], hidden_size)
        # Dropout layer with 30% dropout rate to prevent overfitting
        self.dropout1 = nn.Dropout(0.3)
        # Second fully connected layer: Maps hidden layer to output (binary classification: 2 classes)
        self.fc2 = nn.Linear(hidden_size, 2)
        # LogSoftmax activation for output, gives log-probabilities over the two classes
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        # Apply the embedding layer to convert indices to embeddings
        x = self.embedding(x)
        # Perform mean pooling to average the word embeddings in the sentence
        x = torch.mean(x, dim=1)
        # Pass the pooled embeddings through the first fully connected layer and apply ReLU activation
        x = F.relu(self.fc1(x))
        # Apply dropout to the activations to reduce overfitting
        x = self.dropout1(x)
        # Pass through the second fully connected layer, then apply ReLU activation
        x = F.relu(self.fc2(x))
        # Apply LogSoftmax to get log-probabilities for each class
        x = self.log_softmax(x)
        return x

# A DAN with 2 (trainable) fully-connected layers, and embedding, dropout, and softmax layers
# uses cross entropy loss function with ReLu Activation Functions
class NNDAN2(nn.Module):
    def __init__(self, embedding_matrix, hidden_size, fine_tune):
        super().__init__()
        # Embedding layer using pre-trained GloVe embeddings
        # If fine_tune is False, the embedding weights are frozen
        self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(embedding_matrix), freeze=not fine_tune)
        # First fully connected layer: Transforms from embedding size to hidden size
        self.fc1 = nn.Linear(embedding_matrix.shape[1], hidden_size)
        # Dropout layer with 30% dropout rate to prevent overfitting
        self.dropout1 = nn.Dropout(0.3)
        # Second fully connected layer: Maps hidden layer to output (binary classification: 2 classes)
        self.fc2 = nn.Linear(hidden_size, 2)

    def forward(self, x):
        # Apply the embedding layer to convert indices to embeddings
        x = self.embedding(x)
        # Perform mean pooling to average the word embeddings in the sentence
        x = torch.mean(x, dim=1)
        # Pass the pooled embeddings through the first fully connected layer and apply ReLU activation
        x = F.relu(self.fc1(x))
        # Apply dropout to the activations to reduce overfitting
        x = self.dropout1(x)
        # Pass through the second fully connected layer, then apply ReLU activation
        x = F.relu(self.fc2(x))
        return x

# A DAN with 2 (trainable) fully-connected layers, and embedding, dropout, and softmax layers
# uses negative log likelihood loss function with tanh and sigmoid activation functions
class NNDAN3(nn.Module):
    def __init__(self, embedding_matrix, hidden_size, fine_tune):
        super().__init__()
        # Embedding layer using pre-trained GloVe embeddings
        # If fine_tune is False, the embedding weights are frozen
        self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(embedding_matrix), freeze=not fine_tune)
        # First fully connected layer: Transforms from embedding size to hidden size
        self.fc1 = nn.Linear(embedding_matrix.shape[1], hidden_size)
        # Dropout layer with 30% dropout rate to prevent overfitting
        self.dropout1 = nn.Dropout(0.3)
        # Second fully connected layer: Maps hidden layer to output (binary classification: 2 classes)
        self.fc2 = nn.Linear(hidden_size, 2)
        # LogSoftmax activation for output, gives log-probabilities over the two classes
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        # Apply the embedding layer to convert indices to embeddings
        x = self.embedding(x)
        # Perform mean pooling to average the word embeddings in the sentence
        x = torch.mean(x, dim=1)
        # Pass the pooled embeddings through the first fully connected layer and apply ReLU activation
        x = F.tanh(self.fc1(x))
        # Apply dropout to the activations to reduce overfitting
        x = self.dropout1(x)
        # Pass through the second fully connected layer, then apply ReLU activation
        x = F.sigmoid(self.fc2(x))
        # Apply LogSoftmax to get log-probabilities for each class
        x = self.log_softmax(x)
        return x

# A DAN with 3 (trainable) fully-connected layers, and embedding, 2 dropout layers, and a softmax layer
# uses negative log likelihood loss function and AdamW optimizer
class NNDAN4(nn.Module):
    def __init__(self, embedding_matrix, hidden_size, fine_tune):
        super().__init__()
        # Embedding layer using pre-trained GloVe embeddings
        # If fine_tune is False, the embedding weights are frozen
        self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(embedding_matrix), freeze=not fine_tune)
        # First fully connected layer: Transforms from embedding size to hidden size
        self.fc1 = nn.Linear(embedding_matrix.shape[1], hidden_size)
        # Dropout layer with 30% dropout rate to prevent overfitting
        self.dropout1 = nn.Dropout(0.3)
        # Second fully connected layer: Maps hidden layer to hidden layer
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        # Dropout layer with 50% dropout rate to prevent overfitting
        self.dropout2 = nn.Dropout(0.5)
        # Third fully connected layer: Maps hidden layer to output (binary classification: 2 classes)
        self.fc3 = nn.Linear(hidden_size, 2)
        # LogSoftmax activation for output, gives log-probabilities over the two classes
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        # Apply the embedding layer to convert indices to embeddings
        x = self.embedding(x)
        # Perform mean pooling to average the word embeddings in the sentence
        x = torch.mean(x, dim=1)
        # Pass the pooled embeddings through the first fully connected layer and apply tanh activation
        x = F.tanh(self.fc1(x))
        # Apply dropout to the activations to reduce overfitting
        x = self.dropout1(x)
        # Pass through the second fully connected layer, then apply tanh activation
        x = F.tanh(self.fc2(x))
        # Apply dropout to the activations to reduce overfitting
        x = self.dropout1(x)
        # Pass through the third fully connected layer, then apply Sigmoid activation
        x = F.sigmoid(self.fc3(x))
        # Apply LogSoftmax to get log-probabilities for each class
        x = self.log_softmax(x)
        return x

# A DAN with 3 (trainable) fully-connected layers, and random weight embedding, 2 dropout layers, and a softmax layer
# uses negative log likelihood loss function and AdamW optimizer
class NNDAN5(nn.Module):
    def __init__(self, embedding_matrix, hidden_size, fine_tune):
        super().__init__()
        # Embedding layer using random weights
        self.embedding = embedding_matrix
        # First fully connected layer: Transforms from embedding size to hidden size
        self.fc1 = nn.Linear(embedding_matrix.weight.shape[1], hidden_size)
        # Dropout layer with 30% dropout rate to prevent overfitting
        self.dropout1 = nn.Dropout(0.3)
        # Second fully connected layer: Maps hidden layer to hidden layer
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        # Dropout layer with 50% dropout rate to prevent overfitting
        self.dropout2 = nn.Dropout(0.5)
        # Third fully connected layer: Maps hidden layer to output (binary classification: 2 classes)
        self.fc3 = nn.Linear(hidden_size, 2)
        # LogSoftmax activation for output, gives log-probabilities over the two classes
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        # Apply the embedding layer to convert indices to embeddings
        x = self.embedding(x)
        # Perform mean pooling to average the word embeddings in the sentence
        x = torch.mean(x, dim=1)
        # Pass the pooled embeddings through the first fully connected layer and apply tanh activation
        x = F.tanh(self.fc1(x))
        # Apply dropout to the activations to reduce overfitting
        x = self.dropout1(x)
        # Pass through the second fully connected layer, then apply tanh activation
        x = F.tanh(self.fc2(x))
        # Apply dropout to the activations to reduce overfitting
        x = self.dropout1(x)
        # Pass through the third fully connected layer, then apply Sigmoid activation
        x = F.sigmoid(self.fc3(x))
        # Apply LogSoftmax to get log-probabilities for each class
        x = self.log_softmax(x)
        return x


##############################################################################
# PART 2
##############################################################################

# A DAN with 3 (trainable) fully-connected layers, and random weight embedding, 2 dropout layers, and a softmax layer
# uses negative log likelihood loss function and AdamW optimizer
class NNDAN6(nn.Module):
    def __init__(self, vocabsize, vectordim, hidden_size, fine_tune):
        super().__init__()
        # Embedding layer using random weights
        self.embedding = nn.Embedding(vocabsize, vectordim) #shape(|V|, d)
        # First fully connected layer: Transforms from embedding size to hidden size
        self.fc1 = nn.Linear(vectordim, hidden_size)
        # Dropout layer with 30% dropout rate to prevent overfitting
        self.dropout1 = nn.Dropout(0.3)
        # Second fully connected layer: Maps hidden layer to hidden layer
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        # Dropout layer with 50% dropout rate to prevent overfitting
        self.dropout2 = nn.Dropout(0.5)
        # Third fully connected layer: Maps hidden layer to output (binary classification: 2 classes)
        self.fc3 = nn.Linear(hidden_size, 2)
        # LogSoftmax activation for output, gives log-probabilities over the two classes
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        # Apply the embedding layer to convert indices to embeddings
        x = self.embedding(x)
        # Perform mean pooling to average the word embeddings in the sentence
        x = torch.mean(x, dim=1)
        # Pass the pooled embeddings through the first fully connected layer and apply tanh activation
        x = F.tanh(self.fc1(x))
        # Apply dropout to the activations to reduce overfitting
        x = self.dropout1(x)
        # Pass through the second fully connected layer, then apply tanh activation
        x = F.tanh(self.fc2(x))
        # Apply dropout to the activations to reduce overfitting
        x = self.dropout1(x)
        # Pass through the third fully connected layer, then apply Sigmoid activation
        x = F.sigmoid(self.fc3(x))
        # Apply LogSoftmax to get log-probabilities for each class
        x = self.log_softmax(x)
        return x

##############################################################################
# TRAINING, EVALUATION AND FINE-TUNING
##############################################################################

# Training function
def train_epoch(data_loader, model, loss_fn, optimizer):
    # start_time = time.time()
    
    size = len(data_loader.dataset)
    num_batches = len(data_loader)
    model.train()
    train_loss, correct = 0, 0
    for batch, (X, y) in enumerate(data_loader):
        X = X.long()

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

    # end_time = time.time()
    # elapsed_time = end_time - start_time
    # print(f"\nTraining completed in : {elapsed_time} seconds")
    
    return accuracy, average_train_loss


# Evaluation function
def eval_epoch(data_loader, model, loss_fn, optimizer):
    # start_time = time.time()
    
    size = len(data_loader.dataset)
    num_batches = len(data_loader)
    model.eval()
    eval_loss = 0
    correct = 0
    for batch, (X, y) in enumerate(data_loader):
        X = X.long()

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
        eval_loss += loss.item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    average_eval_loss = eval_loss / num_batches
    accuracy = correct / size

    # end_time = time.time()
    # elapsed_time = end_time - start_time
    # print(f"\nEvaluation completed in : {elapsed_time} seconds")
    return accuracy, average_eval_loss


# Experiment function to run training and evaluation for multiple epochs
def experiment(model, train_loader, test_loader, loss_fn, optimizer):
    train_start_time = train_elapsed_time = 0
    test_start_time = test_elapsed_time = 0
    
    if loss_fn == 'NLL':
        loss_fn = nn.NLLLoss()
    else:
        loss_fn = nn.CrossEntropyLoss()

    if optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.01)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        
    all_train_accuracy = []
    all_test_accuracy = []
    for epoch in range(1,101):
        train_start_time = time.time()
        train_accuracy, train_loss = train_epoch(train_loader, model, loss_fn, optimizer)
        all_train_accuracy.append(train_accuracy)
        train_elapsed_time += time.time() - train_start_time

        test_start_time = time.time()
        test_accuracy, test_loss = eval_epoch(test_loader, model, loss_fn, optimizer)
        all_test_accuracy.append(test_accuracy)
        test_elapsed_time += time.time() - test_start_time

        # if epoch % 5 == 0:
        #      print(f'Epoch #{epoch}: train accuracy {train_accuracy:.3f}, test accuracy {test_accuracy:.3f}')

    print(f"\nTraining completed in : {train_elapsed_time} seconds")
    print(f"Evaluation completed in : {test_elapsed_time} seconds\n")
    
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

    #Set up argument parser
    parser = argparse.ArgumentParser(description='Run model training based on specified model type')
    parser.add_argument('--model', type=str, required=True, help='Model type to train (e.g., BOW)')

    #Parse the command-line arguments
    args = parser.parse_args()

    
    ########################################################################################################################
    # PART 1 A
    ########################################################################################################################
    if args.model == "1a" or args.model == "all":
        #Configuration For Tuning
        glovedim=300
        fine_tune=True
        batch_size=16
        vectorizer='glove'
        hidden_size=100
        
        # Load dataset
        start_time = time.time()
    
        train_data = SentimentDatasetDAN("data/train.txt", vectorizer, glovedim=glovedim)
        test_data = SentimentDatasetDAN("data/dev.txt", vectorizer, glovedim=glovedim)
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
        embedding_matrix = train_data.embeddings_matrix # Get the embeddings from the dataset
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Data loaded in : {elapsed_time} seconds")
    
        # Train and evaluate 
        loss_fn = 'NLL'
        optimizer = 'AdamW'
    
        # Changing activation function
        print('\n-----------------------------------------------------------------------------------------------------------') 
        print('\nComparing performance of Neural Networks with 2 fully connected layers, and embedding, and dropout layers') 
        print('\nUsing Negative log likelihood loss function with ReLu')
        nn1_train_accuracy, nn1_test_accuracy = experiment(NNDAN1(embedding_matrix, hidden_size=hidden_size, fine_tune=fine_tune), train_loader, test_loader, loss_fn, optimizer)
        print('\nUsing Negative log likelihood loss function with tanh + sigmoid')
        nn3_train_accuracy, nn3_test_accuracy = experiment(NNDAN3(embedding_matrix, hidden_size=hidden_size, fine_tune=fine_tune), train_loader, test_loader, loss_fn, optimizer)
    
        plot_accuracy({'accuracy': nn1_train_accuracy, 'label': 'ReLu'}, 
                      {'accuracy': nn3_train_accuracy, 'label': 'tanh + sigmoid'},
                     type='train', title='Training Accuracy for 2 Layer Networks using NLL Loss',
                     file='Figure1-train.png')
        
        plot_accuracy({'accuracy': nn1_test_accuracy, 'label': 'ReLu'}, 
                      {'accuracy': nn3_test_accuracy, 'label': 'tanh + sigmoid'},
                     type='test', title='Test Accuracy for 2 Layer Networks using NLL Loss',
                     file='Figure1-test.png')
    
        # Changing loss function
        print('\n-----------------------------------------------------------------------------------------------------------') 
        print('\nComparing performance of Neural Networks with 2 fully connected layers, and embedding, and dropout layers') 
        print('\nUsing ReLu Activation functions and NLL')
        nn1_train_accuracy, nn1_test_accuracy = experiment(NNDAN1(embedding_matrix, hidden_size=hidden_size, fine_tune=fine_tune), train_loader, test_loader, loss_fn, optimizer)
        print('\n\nUsing ReLu Activation functions and Cross Entropy')
        loss_fn = 'CrossEntropy'
        nn2_train_accuracy, nn2_test_accuracy = experiment(NNDAN2(embedding_matrix, hidden_size=hidden_size, fine_tune=fine_tune), train_loader, test_loader, loss_fn, optimizer)
    
        plot_accuracy({'accuracy': nn1_train_accuracy, 'label': 'NLL'}, 
                      {'accuracy': nn2_train_accuracy, 'label': 'Cross Entropy'},
                     type='train', title='Training Accuracy for 2 Layer Networks using ReLu Activation',
                     file='Figure2-train.png')
        
        plot_accuracy({'accuracy': nn1_test_accuracy, 'label': 'NLL'}, 
                      {'accuracy': nn2_test_accuracy, 'label': 'Cross Entropy'},
                     type='test', title='Test Accuracy for 2 Layer Networks using ReLu Activation',
                     file='Figure2-test.png')
    
        # 2 vs 3 Layer with different activation functions
        print('\n-----------------------------------------------------------------------------------------------------------') 
        loss_fn = 'NLL'
        optimzer = 'AdamW'
        print('\nComparing performance of Neural Networks with 2 vs 3 fully connected layers') 
        print('\nUsing 2 Layer NN with ReLu and 1 dropout layer')
        nn1_train_accuracy, nn1_test_accuracy = experiment(NNDAN1(embedding_matrix, hidden_size=hidden_size, fine_tune=fine_tune), train_loader, test_loader, loss_fn, optimizer)
        print('\nUsing 3 layer NN with tanh, sigmoid and 2 dropout layers')
        nn4_train_accuracy, nn4_test_accuracy = experiment(NNDAN4(embedding_matrix, hidden_size=hidden_size, fine_tune=fine_tune), train_loader, test_loader, loss_fn, optimizer)
    
        plot_accuracy({'accuracy': nn1_train_accuracy, 'label': '2 Layer'}, 
                      {'accuracy': nn4_train_accuracy, 'label': '3 Layer'},
                     type='train', title='Training Accuracy for 2 vs 3 Layer NN',
                     file='Figure3-train.png')
        
        plot_accuracy({'accuracy': nn1_test_accuracy, 'label': '2 Layer'}, 
                      {'accuracy': nn4_test_accuracy, 'label': '3 Layer'},
                     type='test', title='Test Accuracy for 2 vs 3 Layer NN',
                     file='Figure3-test.png')
    
        # 3 vs 3 (AdamW v/s SGDwM)
        glovedim=300
        fine_tune=True
        vectorizer='glove'
        
        print('\n-----------------------------------------------------------------------------------------------------------') 
        print('\nSince the best performance is observed for 3 Layer NN. We attempt further fine-tuning on it to see if we can get better performance')
        print('\nComparing performance of 3 layer Neural Networks') 
        
        loss_fn = 'NLL'
        optimzer = 'AdamW'
        
        print('\nUsing 16 batch size, 100 units per hidden layer and AdamW optimizer with NLL')
        nn5_train_accuracy, nn5_test_accuracy = experiment(NNDAN4(embedding_matrix, hidden_size=hidden_size, fine_tune=fine_tune), train_loader, test_loader, loss_fn, optimizer)
        
        batch_size=32 # 32 observations per batch
        hidden_size=128 # increased hidden units to 128
        loss_fn = 'NLL' # Negative Log Likelihood Loss
        optimizer = 'SGDwM' # Stochastic Gradient Descent with Momentum
        
        print('\nUsing 32 batch size, 128 units per hidden layer and SGD with momentum optimizer with NLL')
        nn6_train_accuracy, nn6_test_accuracy = experiment(NNDAN4(embedding_matrix, hidden_size=hidden_size, fine_tune=fine_tune), train_loader, test_loader, loss_fn, optimizer)
    
        plot_accuracy({'accuracy': nn5_train_accuracy, 'label': 'AdamW + 100 Units'}, 
                      {'accuracy': nn6_train_accuracy, 'label': 'SGDwM + 128 Units'},
                     type='train', title='Training Accuracy for 3 vs 3 Layer NN',
                     file='Figure4-train.png')
        
        plot_accuracy({'accuracy': nn5_test_accuracy, 'label': 'AdamW + 100 Units'}, 
                      {'accuracy': nn6_test_accuracy, 'label': 'SGDwM + 128 Units'},
                     type='test', title='Test Accuracy for 3 vs 3 Layer NN',
                     file='Figure4-test.png')

    
    ########################################################################################################################
    # PART 1 B
    ########################################################################################################################
    elif args.model == "1b" or args.model == "all":
        print('\n-----------------------------------------------------------------------------------------------------------') 
        print('\nComparing performance of the 3 Layer NN with fine-tuning pretrained GloVe embeddings vs training Random Embeddings from scratch')
        #Configuration For Tuning
        glovedim=300
        fine_tune=True
        batch_size=16
        vectorizer='random'
        hidden_size=100
    
        loss_fn = 'NLL'
        optimizer = 'AdamW'
    
        # Use Glove Vectorizer for embeddings
        vectorizer='glove'
        
        # Load dataset
        start_time = time.time()
    
        train_data = SentimentDatasetDAN("data/train.txt", vectorizer, glovedim=glovedim)
        test_data = SentimentDatasetDAN("data/dev.txt", vectorizer, glovedim=glovedim)
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
        embedding_matrix = train_data.embeddings_matrix # Get the embeddings from the dataset
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Data loaded in : {elapsed_time} seconds")
    
        print('\nTraining and Evaluating a 3 Layer Neural Network on Pre-trained GloVe Embeddings')
        nn7_train_accuracy, nn7_test_accuracy = experiment(NNDAN4(embedding_matrix, hidden_size=hidden_size, fine_tune=fine_tune), train_loader, test_loader, loss_fn, optimizer)
    
        # Switch Vectorizer to random
        vectorizer='random'
    
        start_time = time.time()
    
        train_data = SentimentDatasetDAN("data/train.txt", vectorizer, glovedim=glovedim)
        test_data = SentimentDatasetDAN("data/dev.txt", vectorizer, glovedim=glovedim)
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
        embedding_matrix = train_data.embeddings_matrix # Get the embeddings from the dataset
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Data loaded in : {elapsed_time} seconds")
    
        print('\nTraining (from scratch) and Evaluating a 3 Layer Neural Network on random weight embeddings')
        nn8_train_accuracy, nn8_test_accuracy = experiment(NNDAN5(embedding_matrix, hidden_size=hidden_size, fine_tune=fine_tune), train_loader, test_loader, loss_fn, optimizer)
    
        plot_accuracy({'accuracy': nn7_train_accuracy, 'label': 'GloVe'}, 
                      {'accuracy': nn8_train_accuracy, 'label': 'Random'},
                     type='train', title='Training Accuracy for GloVe vs Random',
                     file='Figure5-train.png')
        
        plot_accuracy({'accuracy': nn7_test_accuracy, 'label': 'GloVe'}, 
                      {'accuracy': nn8_test_accuracy, 'label': 'Random'},
                     type='test', title='Test Accuracy for GloVe vs Random',
                     file='Figure5-test.png')
    
    ########################################################################################################################
    # PART 2 A
    ########################################################################################################################
    if args.model == "2a" or args.model == "all":
        
        print('\n-----------------------------------------------------------------------------------------------------------') 
        print('Deep Averaging Network (DAN) with subword tokenization using Byte Pair Encoding:')
        #Configuration For Tuning
        vectordim=300
        fine_tune=True
        batch_size=16
        hidden_size=100
        loss_fn = 'NLL'
        optimizer = 'AdamW'
        bpe_merges = 1500

        # Build a subword vocabulary using BPE algorithm
        training_file = 'data/train.txt'
        subword_vocab = build_bpe_subword_vocab(training_file, bpe_merges)
        
        
        # Load dataset
        start_time = time.time()
    
        print('----Loading Training Data----')
        train_data = SentimentDatasetDANWithBPE("data/train.txt", subword_vocab, vectordim=vectordim)
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        print('----Loading Test Data--------')
        test_data = SentimentDatasetDANWithBPE("data/dev.txt", subword_vocab, vectordim=vectordim)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
        vocabsize = len(subword_vocab)
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"\nData loaded in : {elapsed_time} seconds")
    
    
        print('\nTraining (from scratch) and Evaluating a 3 Layer Neural Network using BPE Subword-Level tokenization on random weight embeddings')
        nn9_train_accuracy, nn9_test_accuracy = experiment(NNDAN6(vocabsize=vocabsize, vectordim=vectordim, hidden_size=hidden_size, fine_tune=fine_tune), train_loader, test_loader, loss_fn, optimizer)
    
        # Switch Vectorizer to random
        vectorizer='random'
    
        start_time = time.time()
    
        # print('----Loading Training Data----')
        train_data = SentimentDatasetDAN("data/train.txt", vectorizer, glovedim=vectordim)
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        # print('----Loading Test Data--------')
        test_data = SentimentDatasetDAN("data/dev.txt", vectorizer, glovedim=vectordim)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
        
        embedding_matrix = train_data.embeddings_matrix # Get the embeddings from the dataset
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Data loaded in : {elapsed_time} seconds")
    
        print('\nTraining (from scratch) and Evaluating a 3 Layer Neural Network using Word-Level tokenization on random weight embeddings')
        nn10_train_accuracy, nn10_test_accuracy = experiment(NNDAN5(embedding_matrix, hidden_size=hidden_size, fine_tune=fine_tune), train_loader, test_loader, loss_fn, optimizer)
    
        plot_accuracy({'accuracy': nn9_train_accuracy, 'label': 'Subword (BPE) Tokenization'}, 
                      {'accuracy': nn10_train_accuracy, 'label': 'Word Tokenization'},
                     type='train', title='Training Accuracy for Subword-Level (BPE) vs Word-Level',
                     file='Figure6-train.png')
        
        plot_accuracy({'accuracy': nn9_test_accuracy, 'label': 'Subword (BPE) Tokenization'}, 
                      {'accuracy': nn10_test_accuracy, 'label': 'Word Tokenization'},
                     type='test', title='Test Accuracy for Subword-Level (BPE) vs Word-Level',
                     file='Figure6-test.png')
        
        print('\n\nEvaluating effect of vocabulary length on Accuracy:')
        #Configuration For Tuning
        vectordim=300
        fine_tune=True
        batch_size=16
        hidden_size=100
        loss_fn = 'NLL'
        optimizer = 'AdamW'
        # Varying merges from 100 to 1000
    
        peak_test_accuracy = []
        peak_train_accuracy = []
        vocab_sizes = []
        for bpe_merges in range(500, 5001, 500):
            # Build a subword vocabulary using BPE algorithm
            training_file = 'data/train.txt'
            subword_vocab = build_bpe_subword_vocab(training_file, bpe_merges)
            # Load the data
            # print('----Loading Training Data----')
            train_data = SentimentDatasetDANWithBPE("data/train.txt", subword_vocab, vectordim=vectordim)
            train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
            # print('----Loading Test Data--------')
            test_data = SentimentDatasetDANWithBPE("data/dev.txt", subword_vocab, vectordim=vectordim)
            test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
            vocabsize = len(subword_vocab)
    
            print('\nTraining (from scratch) and Evaluating a 3 Layer Neural Network using BPE (training) vocabulary size: ', vocabsize)
            train_accuracy, test_accuracy = experiment(NNDAN6(vocabsize=vocabsize, vectordim=vectordim, hidden_size=hidden_size, fine_tune=fine_tune), train_loader, test_loader, loss_fn, optimizer)
            print('Peak Train Accuracy achieved: ', max(train_accuracy))
            print('Peak Test Accuracy achieved: ', max(test_accuracy))
            peak_test_accuracy.append(max(test_accuracy))
            peak_train_accuracy.append(max(train_accuracy))
            vocab_sizes.append(vocabsize)
    
        
        # Plot the training accuracy
        plt.figure(figsize=(8, 6))
        plt.plot(vocab_sizes, peak_test_accuracy, label='Test Accuracy')
        plt.plot(vocab_sizes, peak_train_accuracy, label='Train Accuracy')
        plt.xlabel('Vocabulary Size')
        plt.ylabel('Test Accuracy')
        plt.title('Accuracy for Subword-Level Tokenization(BPE) as a function of Vocabulary Length')
        plt.legend()
        plt.grid()
        plt.savefig('Figure7.png')
        
    
if __name__ == "__main__":
    main()