
# Step 1: Importing of libraries
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
import random

import nltk
nltk.download('punkt')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import json
import pickle

# Step 2: Preprocess Data
words = []
classes = []
documents = []
ignore_letters = ['!', '?', '.', ',' ]
with open('intents.json') as file:
    intents = json.load(file)
# intents_file = open('intents.json').read()
# intents = json.loads(intents_file)

for intent in intents['intents']:
    for pattern in intent['patterns']:
        #tokenize each word
        word = nltk.word_tokenize(pattern)
        words.extend(word)
        #add documents in the corpus
        documents.append((word, intent['tag']))
        #add to classes list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])
            
#print(documents)
# Lemmatize, lower each word and remove duplicates
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_letters]
words = sorted(list(set(words)))
# Sorting of classes
classes = sorted(list(set(classes)))
# documents > combination between patterns and intents
print(len(documents), "documents")
# classes = intents
print(len(classes), "classes", classes)
# words = all words, vocabulary
print(len(words), "unique lemmatized words", words)

# Saving the python object in a file for post-training/prediction 
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# Step 3: Creation of Training and Testing Data
training = []
# empty array for output
output_empty = [0] * len(classes)
# training set, BoW for each sentence
for doc in documents:
    # init BoW
    bag = []
    # list of tokenized words for pattern
    pattern_words = doc[0]
    # lemmatize each word - create base word, to represent related words
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    # create BoW array with 1, if match found in current pattern
    for word in words:
        bag.append(1) if word in pattern_words else bag.append(0)
        
    # output is 0 for each tag and 1 for current tag
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    training.append([bag, output_row])

# shuffle the features and make np.array
random.shuffle(training)
training = np.array(training)
# create train and test lists. X = patterns, Y = intents
train_x = list(training[:, 0])
train_y = list(training[:, 1])
print("Training data created")

# Step 4: Training Model
# model creation - 3 layers (128, 64, number of neurons)
model = Sequential()
model.add(Dense(128, input_shape = (len(train_x[0]),), activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation = 'softmax'))

# compile model, Stochastic gradient descent with Nesterov accelerated gradient
sgd = SGD(lr = 0.01, decay = 1e-6, momentum = 0.9, nesterov = True)
model.compile(loss = 'categorical_crossentropy', optimizer = sgd, metrics = ['accuracy'])

# fitting and saving model
hist = model.fit(np.array(train_x), np.array(train_y), epochs = 200, batch_size = 5, verbose = 1)
model.save('chatbot_model.h5', hist)
print("model created")

# Step 5: Interaction with Chatbot(Proceed to gui_chatbot.py)