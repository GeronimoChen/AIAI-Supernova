import sys

modelposition=sys.argv[1]
print(modelposition)
Xtestposition=sys.argv[2]
Youtposition=sys.argv[3]
observeoutposition=sys.argv[4]


import os
######TheEnvironmentSpecifier######
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

spectralist=[
    'ObserveSpectra/Prediction/SN2011fe/SN2011fe_0.4d.dat',
    'ObserveSpectra/Prediction/SN2011fe/SN2011fe_-2.6d.dat',
    'ObserveSpectra/Prediction/SN2011fe/SN2011fe_3.7d.dat',
    'ObserveSpectra/Prediction/SN2013dy/SN2013dy_-3.1d.flm',
    'ObserveSpectra/Prediction/SN2013dy/SN2013dy_-1.1d.flm',
    'ObserveSpectra/Prediction/SN2013dy/SN2013dy_0.9d.flm',
    'ObserveSpectra/Prediction/SN2013dy/SN2013dy_3.9d.flm',
    'ObserveSpectra/Prediction/SN2011iv/SN2011iv_0.6d.flm',
    'ObserveSpectra/Prediction/SN2015F/SN2015F_-2.3d.flm',
    'ObserveSpectra/Prediction/ASASSN-14lp/ASASSN-14lp_-4.4d.flm',
    'ObserveSpectra/Prediction/SN2011by/SN2011by_-0.4d.flm']
ext1list=[CCM89(Rv=3.1),CCM89(Rv=3.1),CCM89(Rv=3.1),
         CCM89(Rv=3.1),CCM89(Rv=3.1),CCM89(Rv=3.1),CCM89(Rv=3.1),
         CCM89(Rv=3.1),CCM89(Rv=3.1),CCM89(Rv=3.1),F99(Rv=3.1)]
ext2list=[GCC09_MWAvg(),GCC09_MWAvg(),GCC09_MWAvg(),
         GCC09_MWAvg(),GCC09_MWAvg(),GCC09_MWAvg(),GCC09_MWAvg(),
         GCC09_MWAvg(),GCC09_MWAvg(),GCC09_MWAvg(),GCC09_MWAvg()]
ext1EbvList=[0,0,0,0.206,0.206,0.206,0.206,0,0.035,0.33,0.039]
ext2EbvList=[0,0,0,0.135,0.135,0.135,0.135,0,0.175,0.021,0.013]
ElemNameList=[
    'TdCache/ChiMatch/SN2011fe_0.4d.abund.dat',
    'TdCache/ChiMatch/SN2011fe_-2.6d.abund.dat',
    'TdCache/ChiMatch/SN2011fe_3.7d.abund.dat',
    'TdCache/ChiMatch/SN2013dy_-3.1d.abund.dat',
    'TdCache/ChiMatch/SN2013dy_-1.1d.abund.dat',
    'TdCache/ChiMatch/SN2013dy_0.9d.abund.dat',
    'TdCache/ChiMatch/SN2013dy_3.9d.abund.dat',
    'TdCache/ChiMatch/SN2011iv_0.6d.abund.dat',
    'TdCache/ChiMatch/SN2015F_-2.3d.abund.dat',
    'TdCache/ChiMatch/ASASSN-14lp_-4.4d.abund.dat',
    'TdCache/ChiMatch/SN2011by_-0.4d.abund.dat']
zlist=[0.000804,0.000804,0.000804,0.00389,0.00389,0.00389,0.00389,0.006494,0.0049,0.0051,0.002843]



######HereItIsTheSpecifier######



def NewPredictor(spectra,ext1=CCM89(Rv=3.1),ext2=GCC09_MWAvg(),ext1bv=0,ext2bv=0,z=0):
    #To notice, the ext1 is the extinction of the host galaxy, while the ext2 is the extinction of milky way. 
    spectra[:,1]=spectra[:,1]/ext2.extinguish(spectra[:,0]*u.AA,Ebv=ext2bv)
    spectra[:,0]=spectra[:,0]/(1+z)
    spectra[:,1]=spectra[:,1]/ext1.extinguish(spectra[:,0]*u.AA,Ebv=ext1bv)
    fw=interp1d(spectra[:,0],spectra[:,1],fill_value='extrapolate')
    flux=fw(wave)
    spnew=np.array([wave,flux]).T
    spnew=Normalizer(spnew)
    flux=spnew[:,1]
    Yout=model.predict(flux.reshape(1,2000,1))
    return Yout

ObserveYoutData=[]
for i in range(11):
    spr=np.genfromtxt(spectralist[i])
    ObserveYoutData.append(NewPredictor(spr,ext1list[i],ext2list[i],ext1EbvList[i],ext2EbvList[i],zlist[i]))
    
ObserveYoutData=np.array(ObserveYoutData)
np.save(observeoutposition,ObserveYoutData)








