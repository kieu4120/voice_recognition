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
    #print(sample_rate)
    
    mfccs_features = librosa.feature.mfcc(y = audio, sr = sample_rate, n_mfcc = 40)
    mfccs_scale_features = np.mean(mfccs_features.T, axis = 0)
    if(file_name == './Project 2 Database/Jason Truong/Distance1.wav'):
        plt.figure(figsize = (12,4))
        plt.plot(audio)
        plt.plot(mfccs_scale_features)
        
        #show spectrogram
        librosa.display.specshow(mfccs_features, x_axis='time', y_axis='log');
        plt.colorbar(format='%+2.0f dB');
        plt.title('Spectrogram');
        
    return mfccs_scale_features

def features_extraction2(file_name):
    audio, sample_rate = librosa.load(file_name, res_type= 'kaiser_fast')
    #print(sample_rate)
    
    mfccs_features = librosa.feature.melspectrogram(y = audio, sr = sample_rate, n_fft = 2048 )
    mfccs_scale_features = np.mean(mfccs_features.T, axis = 0)
    if(file_name == './Project 2 Database/Jason Truong/Distance1.wav'):
        plt.figure(figsize = (12,4))
        plt.plot(mfccs_scale_features)
        
        #show spectrogram
        librosa.display.specshow(mfccs_features, x_axis='time', y_axis='log');
        plt.colorbar(format='%+2.0f dB');
        plt.title('Spectrogram');
    return mfccs_scale_features

#return: audios and their labels: parameter, dir, to_save, save_dir
def get_audios(audio_directory,to_save = False, save_directory = ''):
    count = 0

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
                    count += 1

                    #---CONVERT AUDIO TO SPECTROGRAM-----
                    #print(file)
                    file_name = './'+ audio_directory +'/' + subfolder + '/' + file
                    
                    data = features_extraction2(file_name)
                    
                    #ENHANCING:                     

                    #append the audio
                    X.append(data)

                    #SUBFOLDERS OR LALBEL,append subfolders
                    y.append(subfolder)
                    
                if count % 50:
                    print('Loaded: ' + str(count) + ' audios')
            
                    
    print("Time for get_image: --- %.2f minutes ---" % ((time.time() - start_time) / 60))
    print("All images are loaded")     
    # return the images and their labels      
    return X, y

