# AIAI-Supernova

It is the code repository for "Artificial Intelligence Assisted Inversion (AIAI) of Synthetic Type Ia Supernovae" (arxiv:1911.05209).  

Because of the data limit in github, I upload the full spectral flux and element abundance data (2 GB) onto kaggle, the website is https://www.kaggle.com/geronimoestellarchen/aiaisupernova . 

## Environment Requirements

-keras  
-tardis  
-tensorflow  
-dust_extinction 0.7+ (https://dust-extinction.readthedocs.io/en/latest/index.html)  


## The Code Structure

Most of the codes are in jupyter notebook, and I don't intend to write "class" in my code, one reason is it let the code less readable, the other is I don't know how to write "classes"...  

The repository size is about 3 GB, including 34 trained neural networks trained on 90000 spectra and 92 trained neural networks trained on 10000 spectra. 
Persumably it will be larger when I upload more neural networks.  

"PrepareTheExecution.ipynb" contains the code about how to automatically write the TARDIS's element abundance files and yaml configuration files.  

"ExecuteCNN.ipynb" contains the code about how to train the neural network for element abundance prediction.  

"ExecutePredict.ipynb" contains the code about how to use the trained neural network for element abundance predictions. Also, it tells how to use the testing dataset to estimate the one-sigma error.  

"ExecuteTardis.ipynb" contains the code about how to insert a observed spectra then get the element abundance predictions from trained neural networks. 
Using these predictions, you can run the TARDIS again and fit the simulation to the observation. 
But still, you need to try several times to determine the best photosphere and requested luminosty.  



## Data Structure

In the "DataSet/", a small portion of the data are given ("X_small.npy", "Y_small.npy"). 
"X_small.npy" contains 1234 spectra, each spectra contains 2000 pixels. 
"Y_small.csv" contains the relating element abundances which used to simulate the 1234 spectra. 

## To-Do Lists

I have not formalized the code for the 3000-5200 Angstrom wavelength spectra, and the code for absolute luminosity predictions. 
Also, I didn't upload the template comparison code to estimate the galactic extinction effect yet. 
Maybe, I can launch a kaggle competition for it? 
Because I am not so sure my neural network structure is the finest-tuned, any weird network structures are welcomed. 