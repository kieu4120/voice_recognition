# voice_recognition

voice classification system using SVC and KNN classifier

In our project we looked three things, two of which we focused on most:
- Distance
- Gender
- Voice isolation


Method: audio preprocessing: 
 - noisereduce
 - MEL-spectrogram 
 - MFCC 

Training and testing: 
- 30% for test and 70% for training 
- Classifiers: SVC, GaussianNB, Neural Network
- Score lelve fusion: get the average matching scores of 3 classifier to create an Optimization System 

Evaluate the system: 
We can evaluate the accuracy of the classifier through the AUC. The more area under the curve, the better the Model for distinguishing between classes.
EER can be obtained based on FAR and FRR. The system with the lowest EER is the most accurate.

