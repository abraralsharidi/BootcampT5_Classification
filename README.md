## Diabetes Classification

# Introduction

In this project and according to the data set, we will use supervised machine learning techniques.
Particularly, classification to predict the outcome of a diabetic patient based on set of features which are thought to be very important 
and which are likely able to explain the variability in disease outcome in different patients.


# Approach

In this project, we are going to propose a set of questions which tailor or guide the overall data analysis approach that will be taken in order to better understand the data and gain more insights from it to instruct future decisions. There are two medically important predictions to be made which most likely are important for both diagnosis and prognosis of diabetes in different patients.

# Questions

1. Is the patient diabetic or not based on a set of important factors such as
   &quot;Serum Glucose level&quot;, &quot;number of diagnoses&quot;, Type of primary diagnosis and secondary diagnoses, history of diabetes related medications, insulin intake and laboratory related procedures?
   
2. Will the patient be readmitted to the hospital again? and If so, 
   what is the expected relapse or readmission of the patient given his/her current representation or state of health?
   
# Feature Engineering

- We have combined the two most important columns that represent the outcome of the patient health given the disease (Diabetes) into a single binary column
  named `ddegree` or the disease degree/level in a patient,
That column was named as `readmit` which combines both binary columns `DiabetesMed` and `readmitted`. The definition of these columns is as follows:
  
|Column Name|Column Definition                                                                 |
|-----------|----------------------------------------------------------------------------------|
|DiabetesMed|Indicates if there was any diabetic medication prescribed. Values: “yes” and “no” |
|Readmitted |30 days, “>30” if the patient was readmitted in more than 30 days, and “No” for no|

- We converted Readmitted from `Nominal` into `binary` by encoding `No=0` and `others=1`. while `DiabetesMed` was encoded as `yes=1` and `no=0`.
Logically, this column will represent the level of Diabetes Milletus in a given patient as `1` if the patient has the disease and frequently visits the hospital as
either `inpatient` or `outpatient` and `0` otherwise.


# Methods

- We have used **4** different supervised machine learning classification models/algorithms to classify our target/response variable, in this case `ddegree`.
The algorithms are:
  - Logistic Regression
  - K-Nearest Neighbour (KNN)
  - Decision Tree
  - Random Forrest
- We splitted the data into Training Data (80%), Validation (10%) and Test Data (10%).
- We utilized some machine learning techniques to select the best hyperparameters for each algorithm based on two 
common and well known algorithms, namely, `GridSearchCV`.
  
# Results

<img src="1.png">
<br/>
<br/>

<img src="2.png">
<br/>
<br/>



Model(s) Accuracies:
--------------------

|Model              |Accuracy|
|-------------------|--------|
|K-Nearest Neighbour (k=3)|0.65|
|Logistic Regression|0.66    |
|Decision Trees     |0.66    |
|Random Forrest (n_estimators=300)|0.68|

We are still working on the project, and We are trying to improve our models, we will use Extreme Gradient boosting machines and see whether this
state-of-the-art algorithm would improve the accuracy any further.

# End User Benefits

1. Early prediction of diabetes in susceptible patients is very important equally for clinicians in diagnosing the disease and for patients to be guided on their treatment plan and government to be able to evaluate and predict the cost on treatments in advance and set up plans and strategies to support patients and their families during the treatment.
2. Early prediction of re-admission is very important not only to patients or clinicians but also to the hospitals themselves, to proactively respond and prepare beds and equipment to treat those patients.


# Flask Web Application (Diabetes Classification):

- Please <a href="https://github.com/Memo0500/classification_app" target="blank">Click here to open the application</a>


# Presentation (Power Point)

- Please <a href="https://github.com/Memo0500/diabetes_classification/blob/master/Demo (Diabetes_Classification).pptx">Click here to download the Power point presentation</a>


# Used Tools:

- Jupyter notebook
- Python

# Used libraries:

- Pandas.
- NumPy.
- SciPy.
- Matplotlib.
- Seaborn.
- SQLite
- Panda profiling.
- Scikit-learn




