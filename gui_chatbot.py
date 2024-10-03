# Step 5: Interaction with Chatbot
# Import libraries
import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np
from keras.models import load_model
model = load_model('chatbot_model.h5')
import json
import random
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl','rb'))
classes = pickle.load(open('classes.pkl','rb'))

def clean_up_sents(sentence):
    # tokenize the pattern - split words into array
    sentence_words = nltk.word_tokenize(sentence)
    # stem all words 
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# return BoW array
def BoW(sentence, words, show_details = True):
    # tokenize patterns
    sentence_words = clean_up_sents(sentence)
    # BoW vocab matrix
    bag = [0] * len(words)
    for s in sentence_words:
        for i, word in enumerate(words):
            if word == s:
                # assign 1 if word is in vocab
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % word)
    return(np.array(bag))

def predict_class(sentence):
    # filter below threshold preds
    preds = BoW(sentence, words, show_details = False)
    res = model.predict(np.array([preds]))[0]
    Err_Threshold = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > Err_Threshold]
    # sort strength probability
    results.sort(key = lambda x: x[1], reverse = True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag'] == tag):
            result = random.choice(i['responses'])
            break
    return result

players = {
    "Lamine Yamal": "Lamine Yamal has played 10 matches, scored 5 goals, and assisted 5 times.",
    "Pablo Martín Páez Gavira": "Pablo Martín Páez Gavira has played 0 matches, scored 0 goals, and assisted 0 times. But it's important to know that he has been injured and is still the baby goat.",
    "Ansu Fatboy Fati": "Ansu Fatboy Fati has played 2 matches, scored 0 goals, and assisted 0 times. Copium.",
    "Pedrinho Lee": "Pedrinho Lee has played 10 matches, scored 2 goals, and assisted 2 times.",
    "Marc Bernal": "Marc Bernal has played 3 matches, scored 0 goals, and assisted 0 times. Currently, he's out for the season and requires baby-like care.",
    "Ferminho": "Ferminho has played 3 matches, scored 0 goals, and assisted 0 times.",
    "Pau Cubarsi": "Pau Cubarsi has played 10 matches, scored 0 goals, and assisted 1 time."
}

def extract_player_name(msg):
    # Normalize the message to lowercase
    msg_lower = msg.lower()
    for player in players.keys():
        # Check if the player's name is in the message
        if player.lower() in msg_lower:
            return player  # Return the player name if found
    return None  # Return None if no player name is found

def get_player_stats(player_name):
    # Check if the player name is in the players dictionary
    if player_name in players:
        return players[player_name]  # Return the player's stats
    else:
        return "Player not found. Please check the name and try again."

# Create tkinter GUI
import tkinter
from tkinter import *

# Defines the send function
def send(event=None):
    msg = EntryBox.get().strip()  # For Entry widget
    EntryBox.delete(0, END)       # Clear Entry widget

    if msg != '':
        ChatBox.config(state=NORMAL)
        ChatBox.insert(END, "You: " + msg + '\n\n')
        ChatBox.config(foreground="#446665", font=("Verdana", 12 )) 

        ints = predict_class(msg)
         # Check for player stats request
        if "stats" in msg.lower():
            # Extract player name from the previous context or directly from the message
            player_name = extract_player_name(msg)  # You may need a function to get the name
            if player_name:
                # Look up the player's stats here, or set a context variable
                res = get_player_stats(player_name)  # This function should return the correct stat response
            else:
                res = "I couldn't find that player. Please try again."
        else:
            res = getResponse(ints, intents)

        ChatBox.insert(END, "Bot: " + res + '\n\n')           
        ChatBox.config(state=DISABLED)
        ChatBox.yview(END)


root = Tk()
root.title("Chatbot")
root.geometry("400x500")
root.resizable(width=FALSE, height=FALSE)

#Create Chat window
ChatBox = Text(root, bd=0, bg="white", height="8", width="50", font="Arial",)
ChatBox.config(state=DISABLED)

#Bind scrollbar to Chat window
scrollbar = Scrollbar(root, command=ChatBox.yview, cursor="heart")
ChatBox['yscrollcommand'] = scrollbar.set

#Create Button to send message
SendButton = Button(root, font=("Verdana",12,'bold'), text="Send", width="12", height=5, bd=0, bg="#f9a602", activebackground="#3c9d9b",fg='#000000', command= send )

#Create the box to enter message
EntryBox = Entry(root, bd=0, bg="white", fg = "black", width=29, font="Arial")
EntryBox.bind("<Return>", send)
EntryBox.focus()

#Place all components on the screen
scrollbar.place(x=376,y=6, height=386)
ChatBox.place(x=6,y=6, height=386, width=370)
EntryBox.place(x=128, y=401, height=50, width=265)
SendButton.place(x=6, y=401, height=50)

root.mainloop()

