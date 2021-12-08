''' Imports '''
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

from sklearn.multiclass import OneVsRestClassifier as ORC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
''' Import classifier '''
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

from python_speech_features import mfcc, logfbank


from scipy.io import wavfile
import tensorflow as tf
from tensorflow.keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, LSTM
from tensorflow.keras.optimizers import Adam
import wandb
from wandb.keras import WandbCallback
from sklearn import metrics
import matplotlib.pyplot as plt
import pandas


import performance
import audio_process

def score_process(matching_scores, y_test):
    gen_scores = []
    imp_scores = []
    for i in range (len(y_test)):
        scores = matching_scores.loc[i]
        mask = scores.index.isin([y_test[i]])
        
        gen_scores.extend(scores[mask])
        imp_scores.extend(scores[~mask])
    
    return gen_scores, imp_scores

#evaluate the accuracy of the classifier
def classifier_accuracy(clf, X_test, y_test, method = ''):
    # make predictions
    yhat = clf.predict(X_test)
    # evaluate predictions
    accuracy = accuracy_score(y_test, yhat)
    print('Accuracy of %s: %.5f' % (method, accuracy))

def classifier_process(clf, count, method = ''):
    #split data to train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 42)
    clf.fit(X_train, y_train)
    
    print('Number of trains:', len(y_train))
    print('Number of tests: ', len(y_test))
    
    #evaluate the accuracy of the classifier
    classifier_accuracy(clf, X_test, y_test, method)
    
    #get the matching score by predict the probability of X test
    matching_scores = clf.predict_proba(X_test)
    
    classes = clf.classes_

    matching_scores = pandas.DataFrame(matching_scores, columns = classes)
    
    gen_scores, imp_scores = score_process(matching_scores, y_test)
    
    count += 1
    
    return gen_scores, imp_scores, matching_scores, count, y_test
#-----MAIN CODE--------
#audio_directory = 'Project 2 Database'
audio_directory = 'Gender Dataset'

#return: audios and their labels: parameter: audio_dir, to_save, save_dir
X, y = audio_process.get_audios(audio_directory,False, 'Data/')

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=20)
'''
print()
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)'''


print('System:')
number_of_classifier = 0
# create an instance of the classifier
clf = ORC(SVC(kernel = 'linear', probability=True))
#clf = ORC(SVC(probability=True))
method = 'SVC'
gen_scores, imp_scores, matching_scores, number_of_classifier, y_test = classifier_process(clf, number_of_classifier, method)

clf2 = ORC(KNeighborsClassifier(3))
method2 = 'KNN(k = 3)'
#clf2 = ORC(GaussianNB())
#method2 = 'Gaussian Naive Bayes'
gen_scores2, imp_scores2, matching_scores2, number_of_classifier, y_test = classifier_process(clf2, number_of_classifier, method2)

clf3 = ORC(MLPClassifier(random_state=1, max_iter=300))
method3 = 'Neural_network'
gen_scores3, imp_scores3, matching_scores3, number_of_classifier, y_test = classifier_process(clf3, number_of_classifier, method3)

#if there are 3 classifiers
if number_of_classifier == 3:
    #apply score_fusion to get the optimal option (get the average)
    matching_scores_avg = (matching_scores + matching_scores2 + matching_scores3) / number_of_classifier
    
    #collect genuine and impostor scores of the optimal case
    gen_scores_avg, imp_scores_avg = score_process(matching_scores_avg, y_test)
    method_avg = 'Score Level Fusion'
    
    #use classifiers for performance
    performance.performance(gen_scores, imp_scores, gen_scores2, imp_scores2, gen_scores3, imp_scores3, 
                            gen_scores_avg, imp_scores_avg ,method, method2, method3, method_avg, 500)




'''
print(len(y))
wandb.init()
config = wandb.config
channels = 1
config.max_len = 11
config.buckets = 20

model = Sequential()
model.add(LSTM(16, input_shape=(config.buckets, config.max_len, channels), activation="sigmoid"))
model.add(Dense(1, activation='sigmoid'))
model.add(Dense(len(y), activation='softmax'))'''








'''
frequency_sampling, audio_signal =  wavfile.read("./Project 2 Database/VanQuangKieuLe/normal1.wav")

#audio_signal = audio_signal[:15000]

features_mfcc = mfcc(audio_signal, frequency_sampling)

print('\nMFCC:\nNumber of windows =', features_mfcc.shape[0])
print('Length of each feature =', features_mfcc.shape[1])



features_mfcc = features_mfcc.T
plt.matshow(features_mfcc)
plt.title('MFCC')

filterbank_features = logfbank(audio_signal, frequency_sampling)

print('\nFilter bank:\nNumber of windows =', filterbank_features.shape[0])
print('Length of each feature =', filterbank_features.shape[1])

filterbank_features = filterbank_features.T
plt.matshow(filterbank_features)
plt.title('Filter bank')
plt.show()
'''