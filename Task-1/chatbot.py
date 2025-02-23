import tkinter as tk
from tkinter import scrolledtext
import random
import re
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Define categories of keywords/phrases the chatbot recognizes
categories = {
    "farewell": ["bye", "goodbye", "exit", "quit"],
    "greeting": ["hello", "hi", "hey"],
    "how_are_you": ["how are you", "how's it going"],
    "name": ["name", "who are you"],
    "thank_you": ["thank you", "thanks"],
    "help": ["help"],
}

# Define corresponding responses for each category
responses = {
    "farewell": ["Goodbye! Have a great day!", "See you later!", "Bye! Take care."],
    "greeting": ["Hi there!", "Hello!", "Hey! Nice to see you."],
    "how_are_you": ["I'm doing well, thank you!", "I'm great, thanks for asking!"],
    "name": ["I'm a simple chatbot built with rule-based responses.", "You can call me Chatbot."],
    "thank_you": ["You're welcome!", "No problem!", "Happy to help!"],
    "help": ["I'm here to chat. You can ask me simple questions or talk about various topics."],
}

# Define responses for questions based on topics
question_responses = {
    "weather": ["I don't have real-time weather information.", "You might want to check a weather app for that."],
    "time": ["I don't have a clock, but you can check your device."],
    "name": ["I'm a chatbot created for this task.", "You can call me Chatbot."],
}

# List of question words to identify questions
question_words = ["what", "where", "when", "why", "how", "who", "which"]

# Function to generate bot response based on user input
def generate_response(user_input):
    user_input = user_input.lower()
    user_input = re.sub(r'[^\w\s]', '', user_input)  # Remove punctuation

    # Check for category matches
    for category, keywords in categories.items():
        if any(keyword in user_input for keyword in keywords):
            return random.choice(responses[category])

    # If no category matches, use NLP
    tokens = nltk.word_tokenize(user_input)
    tagged = nltk.pos_tag(tokens)

    # Check if it's a question
    if tokens and tokens[0] in question_words:
        # Check for specific keywords in the question
        for topic, topic_responses in question_responses.items():
            if any(token in topic for token in tokens):
                return random.choice(topic_responses)
        return "I'm not sure about that."
    else:
        # It's a statement
        return random.choice(["That's interesting!", "Tell me more.", "I see."])

# Function to save the chat to a file
def save_chat(user_input, bot_response):
    with open("Task-1/chat_history.txt", "a") as file:
        file.write("User: " + user_input + "\n")
        file.write("Bot: " + bot_response + "\n\n")

# Tkinter GUI setup
root = tk.Tk()
root.title("Chatbot")

# Create text area for conversation history with scrollbar
chat_area = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=50, height=20)
chat_area.grid(row=0, column=0, columnspan=2, padx=10, pady=10)

# Configure tags for styling user and bot messages
chat_area.tag_configure("user", foreground="blue", font=("Helvetica", 10, "bold"))
chat_area.tag_configure("bot", foreground="green")

# Make chat_area read-only
chat_area.config(state=tk.DISABLED)

# Create entry field for user input
entry_field = tk.Entry(root, width=40)
entry_field.grid(row=1, column=0, padx=10, pady=10)

# Function to insert message into chat_area
def insert_message(message, tag):
    chat_area.config(state=tk.NORMAL)
    chat_area.insert(tk.END, message + "\n", tag)
    chat_area.config(state=tk.DISABLED)
    chat_area.see(tk.END)  # Scroll to the end

# Function to handle send button click
def send_message():
    user_input = entry_field.get().strip()
    if user_input:
        # Insert user message
        insert_message("User: " + user_input, "user")
        
        # Generate and insert bot response
        response = generate_response(user_input)
        insert_message("Bot: " + response, "bot")
        
        # Save chat to file
        save_chat(user_input, response)
        
        # Clear entry field
        entry_field.delete(0, tk.END)

# Create send button
send_button = tk.Button(root, text="Send", command=send_message)
send_button.grid(row=1, column=1, padx=10, pady=10)

# Insert initial bot message
insert_message("Bot: Hi! I'm a simple chatbot. You can talk to me.", "bot")

# Start the Tkinter main loop
root.mainloop()
