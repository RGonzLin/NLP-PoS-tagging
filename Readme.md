# Part of Speech tagging 
The Python script included in this repository contains a class aimed at performing Part of Speech (PoS) tagging. It uses a Long Short-Term Memory (LSTM) network implemented using Keras/TensorFlow to this end.  
A Jupyter Notebook is included in which models for the English, Dutch and Irish languages are created and trained; achieving tagging accuracies of around 90% over the test sets.  
Moreover, the files for the pre-trained models are provided, allowing to easily evaluate their performances over your own test sets. To do this just paste the path of a test set in the corresponding line within the Jupyter Notebook.  
Using this code it is easy to construct your own PoS tagging models for a language of your interest. A host of datasets for many languages can be found on the Universal Dependencies website (https://universaldependencies.org), including the ones used for the construction of the aformentioned models. 

## Key features
* Extract sentences from a CoNLL-U file
* Word tokenization and embedding
* LSTM model creation, training and testing
* Saving and loading models
