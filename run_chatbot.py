# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 15:30:52 2024

@author: allomatik
"""

from transformers import pipeline

# load your fine-tuned model and tokenizer
model_path = "./fine_tuned_model"  # replace with the path to your saved model
chatbot = pipeline("text-generation", model=model_path, tokenizer=model_path)

# test the chatbot
conversation = []

while True:
    user_input = input("You: ")
    conversation.append(f"You: {user_input}")
    
    # format input with context
    context = "\n".join(conversation) + "\nBot:"
    
    response = chatbot(
        context, 
        max_new_tokens=100,  # limit response length
        do_sample=True,
        top_p=0.9,
        temperature=0.8,
        truncation=True
        )
    
    # extract only the new response
    bot_reply = response[0]["generated_text"].split("Bot:")[-1].strip()
    print("Bot:", bot_reply)
    
    conversation.append(f"Bot: {bot_reply}")


