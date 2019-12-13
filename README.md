# AIAI-Supernova

It is the code repository for "Artificial Intelligence Assisted Inversion (AIAI) of Synthetic Type Ia Supernovae" (arxiv:1911.05209).  

Because of the data limit in github, I upload the full spectral flux and element abundance data (2 GB) onto kaggle, the website is https://www.kaggle.com/geronimoestellarchen/aiaisupernova . 

## Environment Requirements

-keras  
-tardis  
-astropy  
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

"PredictorOneElem.py" is for running neural networks to predict the element abundances from input spectra, and the same as "PredictorOneElemReallyRunner.py". 
They can be used by "ExecutePredicy.ipynb", I isolate these two scripts out mainly due to the time of reading a neural network into memory.  

## Data Structure

In the "DataSet/", a small portion of the data are given ("X_small.npy", "Y_small.npy"). 
"X_small.npy" contains 1234 spectra, each spectra contains 2000 pixels. 
"Y_small.csv" contains the relating element abundances which used to simulate the 1234 spectra, its column names are the element number and the zone number, each zones match different speed regions of supernova ejecta. 
"wave.npy" contains the wavelength of 2000 pixels of the spectral flux in "X_small.npy".  

In the "DataCache/2019.3.3/", I stored the testing dataset for the neural networks in "MdSaver/HunKRun/". When you are using the "HunKRun" neural networks (which are the best networks trained on about 90000 spectra), please use the pre-trained testing dataset "Xtest.npy" and "Ytest.csv" and don't extract individual testing set from the whole-data.  

In the "IGmodel/", the base element abundances are given in "Element.dat", the element abundances for spectral simulations are the base element abundaces multiply by the "multiplication factor" stored in "DataSet/Y_small.csv". 
All the spectral simulations use a same density structure (but the real density in the simulation is still changing, because not allowing the element abundances sum to one is effectively changing that), and it is "IGmodel/Density.dat".  
"DD-Sprinter-Lumi8.67.yml" is a example of my configurations of running TARDIS simulation.  

In the "DataProduct/", the yaml file, the element abundance, the TARDIS calculted temperature and the TARDIS calculated spectra for 11 of 2000-10000 Angstrom SNe spectra and 15 of 3000-5200 Angstrom SNe spectra are given, the relating observational spectra are available in the "ObserveSpectra/" directory.  

## Supplementary Pictures for the Paper

In the "DataProduct/PaperPlots/". Apart from the pictures shown on the paper, it also contains the simulated spectra from the "predict median" element; the nickle element abundance versus stretch fitting using emcee; nickle's one-sigma variation on simulated spectra, etc.  

## To-Do Lists

I have not formalized the code for the 3000-5200 Angstrom wavelength spectra, and the code for absolute luminosity predictions. 
Also, I didn't upload the template comparison code to estimate the galactic extinction effect yet. 
Maybe, I can launch a kaggle competition for it? 
Because I am not so sure my neural network structure is the finest-tuned, any weird network structures are welcomed. 