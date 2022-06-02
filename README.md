# age_and_gender_prediction



I. Introduction 
The goal of this capstone project is to develop an application, in our case - webpage, with a help of which we will be able to download a photo of the human on the page, and later as an output the gender and age range will be outputted. The predicted gender may be one of ‘Male’ and ‘Female’, and the predicted age may be one of the following ranges- (0–2), (4–6), (8–12), (15–20), (25–32), (38–43), (48–53), (60–100) .
So the first part of the work is to develop a webpage. 
The second part is with the help of OpenCV to create a classification model of images, which will categorize people by age and gender.  
The third part is to connect those two above mentioned steps - webpage and model. So to include this model into our webpage.
Objectives :-
Detect faces
Classify into Male/Female
Classify into one of the 8 age ranges
Put the results on the image and display it
The Dataset -
For this python project, we’ll use the Adience dataset. This dataset serves as a benchmark for face photos and is inclusive of various real-world imaging conditions like noise, lighting, pose, and appearance. The images have been collected from all over the world.
 
Gender Prediction
They have framed Gender Prediction as a classification problem. The output layer in the gender prediction network is of type softmax with 2 nodes indicating the two classes “Male” and “Female”.
 

Age Prediction
Ideally, Age Prediction should be approached as a Regression problem since we are expecting a real number as the output. However, estimating age accurately using regression is challenging. Even humans cannot accurately predict the age based on looking at a person. However, we have an idea of whether they are in their 20s or in their 30s. Because of this reason, it is wise to frame this problem as a classification problem where we try to estimate the age group the person is in. For example, age in the range of 0-2 is a single class, 4-6 is another class and so on.
The Adience dataset has 8 classes divided into the following age groups [(0 – 2), (4 – 6), (8 – 12), (15 – 20), (25 – 32), (38 – 43), (48 – 53), (60 – 100)]. Thus, the age prediction network has 8 nodes in the final softmax layer indicating the mentioned age ranges.
It should be kept in mind that Age prediction from a single image is not a very easy problem to solve as the perceived age depends on a lot of factors and people of the same age may look pretty different in various parts of the world. 
Coding part
The code can be divided into four parts:
Detect Faces
Detect Gender
Detect Age
Display output
We will use many programming languages: Python, CSS, HTML, XML. 
Detect Face
We will use the DNN Face Detector for face detection. The face detection is done using the function getFaceBox. 
 
Predict Gender
We will load the gender network into memory and pass the detected face through the network. The forward pass gives the probabilities or confidence of the two classes. We take the max of the two outputs and use it as the final gender prediction.
Predict Age
We load the age network and use the forward pass to get the output. Since the network architecture is similar to the Gender Network, we can take the max out of all the outputs to get the predicted age group.
Display Output
We will display the output of the network on the input images and show them using the imshow function.
Results
Our website will be able to predict both Gender and Age to high level of accuracy. 
Summary 
In this project we implemented a method using OpenCV to detect the Gender and Age of a person using their image. 
