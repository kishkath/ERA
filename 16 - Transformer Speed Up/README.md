**Session16: Transformer Architectures & Speeding them Up**: The session describes about the bert, bart, gpt's, techniques of speeding up the transformer run with help of dynamic padding, ocp's, etc..

                             ***Holla !, ceci est un projet de traduction de l'anglais vers le français. Allons-y."**
 
### Session 16 Assignment: 

🔏 Problem Statement:

--------------------

         Pick the "en-fr" dataset from opus_books
         
         Remove all the English sentences with more than 150 "tokens"
         
         Remove all french sentences where len(fetch_sentences) > len(english_sentences) + 10 

         Train yor own transformer (E-D) (do anything you want, use PyTorch, OCP, PS, AMP, etc.). but get loss under 1.8! 

    
💡 Define Problem:
------------------
 Pick up the dataset (english to french translation) and maintain the input data constraints, leading to a model having loss less than **1.8**
 
🚦 Follow-up Process:
-----------------
 The directory structure describes in following way:

    Directory: 
    ---------
    ├── Transformer Speed Up
    │   ├── train.py: contains the script of training module
    │    ├── model.py: contains the script for model with encoder, decoder, head attention architectures built upon transformer.
    │   ├── config.py: contains the config values required for the project
    │   ├── dataset.py: contains the code for retrieving the dataset from hugging face, loading data.
    └── README.md Details about the Process.

  Process:
  -------
  * The process initiates with loading the english to french language translation from **opus-books**
  * link of dataset: https://huggingface.co/datasets/opus_books/tree/main
  * Solution:
  * Starting the run of transformer with speed-up architectures


🔑 Model Architecture:
---------------------
 "Transformer Based Built on encoder, decoder": Session14, 15, 16


🔋 Speedup Techniques: 
-------------------



💊 Model Run Results: 
-------------------


