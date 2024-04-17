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
  
    Processing Epoch 20: 100%|██████████| 996/996 [16:48<00:00,  1.01s/it, loss=2.004]
    --------------------------------------------------------------------------------
    SOURCE: At the slightest indisposition of one of your children, you will no longer see them already in the grave.'
    TARGET: À la moindre indisposition de vos enfants, vous ne les verrez plus dans la tombe.
    PREDICTED: Au moindre bruit d ’ une de vos enfants , vous ne les verrez plus par la tombe .
    --------------------------------------------------------------------------------
    SOURCE: "Let us make another trial," resumed the vagabond.
    TARGET: – Essayons encore une fois », reprit le truand.
    PREDICTED: « un autre essai , reprit l ’ abbé .
    ====================================================================================
    
    Processing Epoch 26: 100%|██████████| 996/996 [16:49<00:00,  1.01s/it, loss=1.708]
    --------------------------------------------------------------------------------
    SOURCE: Two or three distinguished themselves by real talent, and, among these, one named Chazel; but Julien felt himself repelled by them, and they by him.
    TARGET: Deux ou trois se distinguaient par un talent réel et, entre autres, un nommé Chazel ; mais Julien se sentait de l’éloignement pour eux et eux pour lui.
    PREDICTED: Deux ou trois mots se de mauvais talent , et , parmi ces , un poignard ; mais Julien se sentait s ’ éloigner et les par lui .
    --------------------------------------------------------------------------------
    SOURCE: I am for those who save the state," said Bonacieux, emphatically.
    TARGET: Je suis pour ceux qui sauvent l'État», dit avec emphase Bonacieux.
    PREDICTED: Je suis pour ceux qui , dit avec beaucoup de l ’ air .
    =====================================================================================
    
    Processing Epoch 28: 100%|██████████| 996/996 [16:50<00:00,  1.01s/it, loss=1.802]
    ----------------------------------------------------------------------------------
    SOURCE: Mais je l’aurai, docteur ; je vous parie deux contre un que je l’aurai.
    TARGET: I shall have him, Doctor—I'll lay you two to one that I have him.
    PREDICTED: I was the first time ; I offer you a which I was to love .
    --------------------------------------------------------------------------------
    SOURCE: But the day before he lefthe was suddenly quite changed, and much softened.
    TARGET: Mais la veille de son départ il parut soudain trèschangé, très adouci.
    PREDICTED: Mais la veille , il se retrouva tout brusquement , tout à coup , et beaucoup s ' .
    --------------------------------------------------------------------------------
    
    Processing Epoch 29:  62%|██████▏   | 621/996 [10:29<06:19,  1.01s/it, loss=1.730]


