# markdownbot
building a basic ai chatbot finetuned using markdown data and opensource pre-trained LLMs. 
accomplished by running a preprocessing script on a markdown database, chunking each markdown file and compiling them into a single text file. 
this text file is used to finetune a pretrained LLM using the huggingface LLM database and training library (https://huggingface.co/models).
the "run_chatbot.py" file allows the user to interact with the fine tuned LLM via the console.

Suggested Workflow:
 - suggest creating a new project folder to work in. copy your markdown database into this new folder to avoid editing the actual markdown vault.
 - point the "preprocessingScript.py" file at your copied markdown database by editing line 8 and run to produce a single text file for finetuning
 - edit lines 15 and 25 in the "fine_tune_model.py" file to point at your chosen pretrained huggingface model (suggest test running code with a small model like gpt2 at first to avoid long wait times) and run the training script.
 - once training is complete, point the "run_chatbot.py" file at the finetuned model by editing line 11 and run the script to start a conversation (suggest running in an IDE, the while loop has no end condition, will have to manually stop/exit to end each conversation)

Notes: 
 - all of the code here was generated using ChatGPT 4o and little has been done to optmize any further. suggest experimenting with different training and preprocessing conditions.
 - deepfates has a preprocessing script for twitter archives as well here (https://gist.github.com/deepfates/78c9515ec2c2f263d6a65a19dd10162d), bit more sophisticated.

