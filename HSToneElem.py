import sys

modelposition=sys.argv[1]
print(modelposition)
Xtestposition=sys.argv[2]
Youtposition=sys.argv[3]
observeoutposition=sys.argv[4]


import os
os.environ['LD_LIBRARY_PATH']='/home/hulei/anaconda3/pkgs/cudatoolkit-9.2-0/lib'
os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES']=''
from keras import backend as K
import tensorflow as tf
import keras
import keras.backend.tensorflow_backend as KTF
KTF.set_session(tf.Session(config=tf.ConfigProto(device_count={'cpu':1})))



import os
import glob
import yaml
import keras
import astropy
import numpy as np
import pandas as pd
import importlib as imp
import keras.backend as K
import astropy.units as u
import astropy.constants as c
from sklearn.manifold import TSNE
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from keras.models import Sequential,Model
from dust_extinction.averages import GCC09_MWAvg
from sklearn.model_selection import train_test_split
from astropy.modeling.blackbody import blackbody_lambda
from dust_extinction.parameter_averages import CCM89,F99
from keras.layers import Dense,Dropout,LocallyConnected1D,AveragePooling1D
from keras.layers import Input,Conv1D,MaxPooling1D,BatchNormalization,Activation,Add,UpSampling1D

wave=np.genfromtxt('Spec/1/0.txt')[:,0]
wave2=wave[448:1167]
def Normalizer(spec,shortwave=6500,longwave=7500):
    small=np.argmin(abs(spec[:,0]-shortwave))
    long=np.argmin(abs(spec[:,0]-longwave))
    if small<long:spec[:,1]=spec[:,1]/np.average(spec[small:long,1])
    if small>long:spec[:,1]=spec[:,1]/np.average(spec[long:small,1])
    return spec

model=keras.models.load_model(modelposition)
X_test=np.load(Xtestposition)
Yout=model.predict(X_test)
np.save(Youtposition,Yout)

HSTlist=pd.read_csv('HSTplotout/EbvChi2HST/HSTlist_Ebv.csv')

def NewPredictor(spectra,ext1=CCM89(Rv=3.1),ext1bv=0,z=0):
    #To notice, the ext1 is the extinction of the host galaxy, while the ext2 is the extinction of milky way. 
    spectra[:,0]=spectra[:,0]/(1+z)
    spectra[:,1]=spectra[:,1]/ext1.extinguish(spectra[:,0]*u.AA,Ebv=ext1bv)
    fw=interp1d(spectra[:,0],spectra[:,1],fill_value='extrapolate')
    flux=fw(wave2)
    spnew=np.array([wave2,flux]).T
    spnew=Normalizer(spnew,shortwave=3000,longwave=5000)
    flux=spnew[:,1]
    Yout=model.predict(flux.reshape(1,719,1))
    return Yout

ObserveYoutData=[]
for i in range(len(HSTlist)):
    name=glob.glob('ObserveSpectra/HST_UV_raw/'+HSTlist['Name'][i]+'*')[0]
    spr=np.genfromtxt(name)[:,0:2]
    ObserveYoutData.append(NewPredictor(spr,CCM89(Rv=3.1),HSTlist['Select'][i],HSTlist['Redshift'][i]))
    
ObserveYoutData=np.array(ObserveYoutData)
np.save(observeoutposition,ObserveYoutData)








