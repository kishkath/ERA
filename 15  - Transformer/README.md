**Session14-15: Dawn of Transformers**: The session describes about the detailed overview of vannila transformers its architectures, history of transformers, Attentions


>>>>> Give me some attention: *****Attention is all you Need*****
  
 
### Session 15 Assignment: 

🔏 Problem Statement:

--------------------

         Pick the "en-it" dataset from opus_books
         
         Convert the pytorch code to pytorch lightning & train the model for 10 Epochs and achieve the loss of less than 4.

    
💡 Define Problem:
------------------
 Pick up the dataset (english to italian translation) leading the model having loss less than **4**
 
🚦 Follow-up Process:
-----------------
 The directory structure describes in following way:

    Directory: 
    ---------
    ├── Transformer
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

        1. Clone the repo
        2. Move to the directory of (15 - Transformer)
        3. Copy the contents of jupyter notebook, and execute the cells.
        4. For fine-tuning modify the parameters of config.py file.

    ### 11th-epoch-model-pt link: 


🔑 Model Architecture:
---------------------
 "Transformer Based Built on encoder, decoder": Session14, 15, 16


💊 Model Run Results: 
-------------------

    Dataset opus_books downloaded and prepared to /root/.cache/huggingface/datasets/opus_books/en-it/1.0.0/e8f950a4f32dc39b7f9088908216cd2d7e21ac35f893d04d39eb594746af2daf. Subsequent calls will reuse this data.
    Max length of source sentence: 309
    Max length of target sentence: 274

    Processing Epoch 00: 100%|██████████| 4850/4850 [26:07<00:00,  3.09it/s, loss=5.847]
    --------------------------------------------------------------------------------
    SOURCE: How will you answer him?
    TARGET: E come gli risponderete?
    PREDICTED: E tu ?
    --------------------------------------------------------------------------------
    SOURCE: The General A.-de-C. disapproved of the races.
    TARGET: Il grande generale deprecava le corse.
    PREDICTED: Il suo giorno era un po ’ di .
--------------------------------------------------------------------------------


    Processing Epoch 02: 100%|██████████| 4850/4850 [26:10<00:00,  3.09it/s, loss=5.010]
    --------------------------------------------------------------------------------
    SOURCE: A quiet wedding we had: he and I, the parson and clerk, were alone present.
    TARGET: Celebrammo un matrimonio quieto quieto; lui, io, il pastore e il vicario e nessun altro.
    PREDICTED: La signora Fairfax era stata una donna , e mi pareva che mi , e mi pareva di .
    --------------------------------------------------------------------------------
    SOURCE: "You will see her this evening," answered Mrs. Fairfax.
    TARGET: — La vedrete stasera.
    PREDICTED: — Siete molto contenta di voi — disse la signora Fairfax .
--------------------------------------------------------------------------------


    Processing Epoch 04: 100%|██████████| 4850/4850 [26:08<00:00,  3.09it/s, loss=4.090]
    --------------------------------------------------------------------------------
    SOURCE: 'Well, and what did you think about me?
    TARGET: — Ma cosa pensavi mai di me?
    PREDICTED: — E allora che cosa mai mi ha fatto ?
    --------------------------------------------------------------------------------
    SOURCE: We had no idea that she was herself there at the station.
    TARGET: Noi non sapevamo nulla, che lei fosse proprio lì, alla stazione.
    PREDICTED: Non avevamo mai pensato che la barca fosse .
--------------------------------------------------------------------------------

    Processing Epoch 06: 100%|██████████| 4850/4850 [26:13<00:00,  3.08it/s, loss=4.090]
    --------------------------------------------------------------------------------
    SOURCE: I know that my husband's brother is dying, that my husband is going to him, and that I am going with my husband in order...'
    TARGET: Io so che il fratello di mio marito sta per morire e mio marito va da lui e io vado con mio marito per....
    PREDICTED: Io so che mio marito è la mia conoscenza , che egli è necessario andare a mio marito e io devo andare in mio marito .
    --------------------------------------------------------------------------------
    SOURCE: Yes, of course I will go,' he decided, lifting his eyes from the book, and a vivid sense of the joy of seeing her made his face radiant.
    TARGET: Certo che andrò” decise fra sé, sollevando la testa dal libro. E, raffiguratasi la gioia nel vederla, si illuminò nel viso.
    PREDICTED: Sì , sì , andrò a finire — disse , guardando il libro e per un tratto di gioia . — Sì , la sua gioia , la sua gioia , la 
---------------------------------------------------------------------------------


    Processing Epoch 08: 100%|██████████| 4850/4850 [26:12<00:00,  3.08it/s, loss=4.327]
    --------------------------------------------------------------------------------
    SOURCE: 'Well, should there be anything – I shall be at Katavasov's.'
    TARGET: — Allora se succede qualcosa, sono da Katavasov.
    PREDICTED: — Su , via , se ne vado via , allora io verrò .
    --------------------------------------------------------------------------------
    SOURCE: This cost me as much thought as a statesman would have bestowed upon a grand point of politics, or a judge upon the life and death of a man.
    TARGET: Ciò mi diede tanti pensieri quanti ne avrebbe dati ad un magistrato il decidere sopra un punto scabrosissimo di politica, o quanti se ne sarebbe presi un giudice prima di sentenziare su la vita o la morte d’un poveretto.
    PREDICTED: Questo contrasto mi spingeva a pensare che il governo sarebbe stato un uomo d ’ accordo con una gran d ’ un uomo o di vita .
-----------------------------------------------------------------------------------


    Processing Epoch 11: 100%|██████████| 4850/4850 [26:09<00:00,  3.09it/s, loss=2.941]
    --------------------------------------------------------------------------------
    SOURCE: "Edward--my little wife!"
    TARGET: — Dite Edoardo, moglie mia!
    PREDICTED: — Signore , mia moglie !
    --------------------------------------------------------------------------------
    SOURCE: While I walked under the dripping orange-trees of my wet garden, and amongst its drenched pomegranates and pine-apples, and while the refulgent dawn of the tropics kindled round me--I reasoned thus, Jane--and now listen; for it was true Wisdom that consoled me in that hour, and showed me the right path to follow.
    TARGET: "Allora presi una risoluzione mentre passeggiavo sotto i gocciolanti alberi di arancio nel mio giardino bagnato dalla pioggia, fra i melagrani e i pini, intanto che la fulgente rugiada dei tropici scendeva intorno a me. "Così ragionai, e voi ascoltatemi, Jane, perché fu la vera saggezza che mi consolò in quell'ora e mi additò la via da seguire.
    PREDICTED: Mentre io conduceva sotto le colonne del giardino , mi e le che erano , mentre ero , allora , mi , e , e ora che ora .
--------------------------------------------------------------------------------


 🎆 Final Result:
---------------------

* Loss : 2.941 at epoch: 11
