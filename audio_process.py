# Load imports
import os 
import cv2
import time
import librosa
import matplotlib.pyplot as plt
from librosa import display
import numpy as np

def features_extraction(file_name):
    audio, sample_rate = librosa.load(file_name, res_type= 'kaiser_fast')
    if(file_name == './Project 2 Database/tessietruong/Distance1.wav'):
        plt.figure(figsize = (12,4))
        plt.plot(audio)
    mfccs_features = librosa.feature.mfcc(y = audio, sr = sample_rate, n_mfcc = 40)
    mfccs_scale_features = np.mean(mfccs_features.T, axis = 0)
    return mfccs_scale_features
    
#return: audios and their labels: parameter, dir, to_save, save_dir
def get_audios(audio_directory,to_save = False, save_directory = ''):
    img_ct = 0

    #X is array of sound wave 
    X = []
    #y is the sampling rate. 
    y = []
    extensions = ('wav')
    #estimate time process
    start_time = time.time()
    
    #custom size of the image
    #dpi = 500
    #print('Image Size = %d' %(dpi) )
    subfolders = os.listdir(audio_directory)
    for subfolder in subfolders:
        print("Loading audios in %s" % subfolder)
        if os.path.isdir(os.path.join(audio_directory, subfolder)): # only load directories
            subfolder_files = os.listdir(
                    os.path.join(audio_directory, subfolder)
                    )
            for file in subfolder_files:
                if file.endswith(extensions):
                    img_ct += 1

                    #---CONVERT AUDIO TO SPECTROGRAM-----
                    #print(file)
                    file_name = './Project 2 Database/' + subfolder + '/' + file
                    data = features_extraction(file_name)
                    
                    #ENHANCING:                     

                    #append the audio
                    X.append(data)

                    #SUBFOLDERS OR LALBEL,append subfolders
                    y.append(subfolder)
                    
    print("Time for get_image: --- %.2f minutes ---" % ((time.time() - start_time) / 60))
    print("All images are loaded")     
    # return the images and their labels      
    return X, y
