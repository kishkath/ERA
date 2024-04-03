**Session16: Transformer Architectures & Speeding them Up**: The session describes about the bert, bart, gpt's, techniques of speeding up the transformer run with help of dynamic padding, ocp's, etc..

  
>>>> ***Holla !, ceci est un projet de traduction de l'anglais vers le français. Allons-y.*** \

  
 
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
    │   ├── model.py: contains the script for model with encoder, decoder, head attention architectures built upon transformer.
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

  Run1: 
  
    Epochs: 25,
    Technique: Usage of cuda 
    torch.cuda.amp.autocast(enabled=True)

  



💊 Model Run Results: 
-------------------

  weights run1:  https://drive.google.com/drive/folders/1pyE84N-6lpxOZQEYYE3OpAq2gNGe7Fz1?usp=drive_link

  logs: 
  
    Processing Epoch 20: 100%|██████████| 996/996 [09:34<00:00,  1.73it/s, loss=1.941]
    --------------------------------------------------------------------------------
    SOURCE: No doubt the presence of the Nautilus, even more fearsome than itself, and which it couldn't grip with its mandibles or the suckers on its arms.
    TARGET: Sans doute de la présence de ce _Nautilus_, plus formidable que lui, et sur lequel ses bras suceurs ou ses mandibules n'avaient aucune prise.
    PREDICTED: Sans doute la présence du cardinal , plus même que lui - même , et qui ne pouvait tenir sa volonté avec des jésuites ou les bras de son équipage .
    --------------------------------------------------------------------------------
    SOURCE: Satisfied with this discovery which confirmed all his suspicions, Athos returned to the hotel, and found Planchet impatiently waiting for him.
    TARGET: Satisfait de cette découverte qui confirmait tous ses soupçons, Athos revint à l'hôtel et trouva Planchet qui l'attendait avec impatience.
    PREDICTED: Julien de cette découverte qui tous ses soupçons , revint vers l ’ hôtel , et Julien l ’ attendait .


