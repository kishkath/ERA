**Session14-15: Dawn of Transformers**: The session describes about the detailed overview of vannila transformers its architectures, history of transformers, Attentions


>>>>> Give me some attention: ******Attention is all you Need*****
  
 
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


🔑 Model Architecture:
---------------------
 "Transformer Based Built on encoder, decoder": Session14, 15, 16


💊 Model Run Results: 
-------------------

2024-03-23 05:13:35.511143: E external/local_xla/xla/stream_executor/cuda/cuda_dnn.cc:9261] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
2024-03-23 05:13:35.511287: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:607] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
2024-03-23 05:13:35.787479: E external/local_xla/xla/stream_executor/cuda/cuda_blas.cc:1515] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
Using device: cuda
Downloading builder script:
6.08k/? [00:00<00:00, 490kB/s]
Downloading metadata:
161k/? [00:00<00:00, 7.22MB/s]
Downloading and preparing dataset opus_books/en-it (download: 3.14 MiB, generated: 8.58 MiB, post-processed: Unknown size, total: 11.72 MiB) to /root/.cache/huggingface/datasets/opus_books/en-it/1.0.0/e8f950a4f32dc39b7f9088908216cd2d7e21ac35f893d04d39eb594746af2daf...
Downloading data: 100%
3.30M/3.30M [00:01<00:00, 3.37MB/s]
Dataset opus_books downloaded and prepared to /root/.cache/huggingface/datasets/opus_books/en-it/1.0.0/e8f950a4f32dc39b7f9088908216cd2d7e21ac35f893d04d39eb594746af2daf. Subsequent calls will reuse this data.
Max length of source sentence: 309
Max length of target sentence: 274
Processing Epoch 00: 100%|██████████| 4850/4850 [26:07<00:00,  3.09it/s, loss=5.847]
stty: 'standard input': Inappropriate ioctl for device
--------------------------------------------------------------------------------
    SOURCE: How will you answer him?
    TARGET: E come gli risponderete?
 PREDICTED: E tu ?
--------------------------------------------------------------------------------
    SOURCE: The General A.-de-C. disapproved of the races.
    TARGET: Il grande generale deprecava le corse.
 PREDICTED: Il suo giorno era un po ’ di .
--------------------------------------------------------------------------------
/opt/conda/lib/python3.10/site-packages/torchmetrics/utilities/prints.py:62: FutureWarning: Importing `CharErrorRate` from `torchmetrics` was deprecated and will be removed in 2.0. Import `CharErrorRate` from `torchmetrics.text` instead.
  _future_warning(
/opt/conda/lib/python3.10/site-packages/torchmetrics/utilities/prints.py:62: FutureWarning: Importing `WordErrorRate` from `torchmetrics` was deprecated and will be removed in 2.0. Import `WordErrorRate` from `torchmetrics.text` instead.
  _future_warning(
/opt/conda/lib/python3.10/site-packages/torchmetrics/utilities/prints.py:62: FutureWarning: Importing `BLEUScore` from `torchmetrics` was deprecated and will be removed in 2.0. Import `BLEUScore` from `torchmetrics.text` instead.
  _future_warning(
Processing Epoch 01: 100%|██████████| 4850/4850 [26:10<00:00,  3.09it/s, loss=5.692]
stty: 'standard input': Inappropriate ioctl for device
--------------------------------------------------------------------------------
    SOURCE: His farming calculations that there is a price below which certain grain must not be sold were forgotten too.
    TARGET: Il calcolo economico che c’era un certo prezzo al di sotto del quale non si poteva vendere una certa qualità di grano, anch’esso era stato dimenticato.
 PREDICTED: Il suo amore è un uomo che non si può fare .
--------------------------------------------------------------------------------
    SOURCE: There was a warm steaming smell of manure when the frozen door opened, and the cows, astonished at the unaccustomed light of the lantern, began moving on their clean straw.
    TARGET: Quando si aprì la porta coperta di gelo, si sentì una zaffata di letame caldo, fumante e le mucche, sorprese dalla luce insolita della lanterna, si agitarono sulla paglia fresca.
 PREDICTED: Era un ’ altra volta , e il suo viso , e il capo , e , , e , la testa , e la testa , , e la testa .
--------------------------------------------------------------------------------
Processing Epoch 02: 100%|██████████| 4850/4850 [26:10<00:00,  3.09it/s, loss=5.010]
stty: 'standard input': Inappropriate ioctl for device
--------------------------------------------------------------------------------
    SOURCE: A quiet wedding we had: he and I, the parson and clerk, were alone present.
    TARGET: Celebrammo un matrimonio quieto quieto; lui, io, il pastore e il vicario e nessun altro.
 PREDICTED: La signora Fairfax era stata una donna , e mi pareva che mi , e mi pareva di .
--------------------------------------------------------------------------------
    SOURCE: "You will see her this evening," answered Mrs. Fairfax.
    TARGET: — La vedrete stasera.
 PREDICTED: — Siete molto contenta di voi — disse la signora Fairfax .
--------------------------------------------------------------------------------
Processing Epoch 03: 100%|██████████| 4850/4850 [26:09<00:00,  3.09it/s, loss=4.677]
stty: 'standard input': Inappropriate ioctl for device
--------------------------------------------------------------------------------
    SOURCE: He read the letters, one of which impressed him unpleasantly.
    TARGET: Lesse le lettere.
 PREDICTED: Egli si mise a parlare con un sorriso che si mise a guardare il suo sorriso .
--------------------------------------------------------------------------------
    SOURCE: This man in a short time restored peace and unity with the greatest success.
    TARGET: Costui in poco tempo la ridusse pacifica et unita, con grandissima reputazione.
 PREDICTED: Questo uomo , in un ’ altra parte , la sua vita , la quale si la sua vita .
--------------------------------------------------------------------------------
Processing Epoch 04: 100%|██████████| 4850/4850 [26:08<00:00,  3.09it/s, loss=4.090]
stty: 'standard input': Inappropriate ioctl for device
--------------------------------------------------------------------------------
    SOURCE: 'Well, and what did you think about me?
    TARGET: — Ma cosa pensavi mai di me?
 PREDICTED: — E allora che cosa mai mi ha fatto ?
--------------------------------------------------------------------------------
    SOURCE: We had no idea that she was herself there at the station.
    TARGET: Noi non sapevamo nulla, che lei fosse proprio lì, alla stazione.
 PREDICTED: Non avevamo mai pensato che la barca fosse .
--------------------------------------------------------------------------------
Processing Epoch 05: 100%|██████████| 4850/4850 [26:12<00:00,  3.08it/s, loss=4.793]
stty: 'standard input': Inappropriate ioctl for device
--------------------------------------------------------------------------------
    SOURCE: I mean, that human affections and sympathies have a most powerful hold on you.
    TARGET: Voglio dire che le affezioni e le simpatie umane esercitano molto potere su di voi.
 PREDICTED: Vi domando che , per questo mondo , sono stati stati .
--------------------------------------------------------------------------------
    SOURCE: Levin knew his brother and the direction of his thoughts, knew that he had become a sceptic not because it was easier for him to live without faith, but because step by step modern scientific explanations of the phenomena of the universe had driven out his faith; he knew therefore that this return to the old faith was not legitimate, not a similar result of thought, but was only a temporary, selfish and irrational hope of recovery.
    TARGET: Levin conosceva il fratello e il corso dei suoi pensieri; sapeva che la sua mancanza di fede era sorta non perché gli fosse più facile vivere senza una fede, ma perché di volta in volta le spiegazioni modernamente scientifiche dei fenomeni del mondo avevano soppiantato la fede, e perciò sapeva che questo suo ritorno alla fede non era legittimo, non era compiuto attraverso lo stesso pensiero, ma era soltanto momentaneo, interessato, per una folle speranza di guarigione.
 PREDICTED: Levin sapeva che il fratello e il fratello , che , malgrado il suo lavoro , non era possibile , ma perché non era possibile , perché , malgrado il suo lavoro , non solo per lui il suo compagno , e non solo per lui stesso , malgrado il suo compagno , malgrado il suo compagno , non solo per lui stesso , ma non solo per lui stesso , non solo per quanto si poteva essere .
--------------------------------------------------------------------------------



