# voice_recognition

voice classification system using SVC and KNN classifier

#research question 
- Using different landmarks, how well can specific facial features be used within a facial recognition system?
- How can we make sure lighting has a minimal effect on how well our facial recognition software works?
- How does the size of the images affect the performance of the system? 

**dataset**
One of the first steps in this project was to determine how big our dataset should be
Our database has 36/50 students from the class
Each student had 43 images 
1548 images total
We didn’t use all 50 students due to processing time
Too big = Longer processing time
Too small = Less accurate reading on our system

**preprocessing: **
We’re testing different sizes of images to find out which one is the best for the detector to recognize the faces.
Detector(HOG) inside OpenCV library
Result: The detector works best when using the original image size

Additionally, We also turn the Images colors into grayscale:
 Using grayscale images yields better results in object recognition*. Grayscale image only has one color channel which easier to process. But you still can obtain the features relating to brightness , contrast, edge, shape, textures without color.
 Using cvtColor from OpenCV (img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
 
 Increasing contrast for dark images: 
Since there are photos with poor lighting conditions (photos with L_D prefix). We decided to increase their contrast using equalizeHist function from OpenCV library. 
The function increases the global contrast of the image. 

We also used Adaptive Histogram Equalization(CLAHE): OpenCV library function that also increase the image contrast.
 Furthermore, it also improves the local contrast and enhances the definition of edges in an image.
 All of the Images are being applied this function

Run an experiment to see how can the preprocessing helps with the system: running 2 images enhancement mention above to many datasets varied in images’ size.

Result: slight increment from 2.3 to 4.4 percents

#Detecting Facial Features Using Landmarks
To answer our first research question we used sets of landmarks to identify specific features within the face.
The different landmark locations for our system range from 0 - 67
As you can see from the image if we wanted to select the eyes as our test facial feature we need to use the landmark range from [36,47]
To do this we just change the range of the for loop within the shape_to_np function.






