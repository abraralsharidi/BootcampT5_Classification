#
# **Classification on Diabetes**

# Introduction

It is increasingly recognized that the management of hyperglycemia in the hospitalized patient has a significant bearing on outcome, in terms of both morbidity and mortality. This recognition has led to the development of formalized protocols in the intensive care unit (ICU) setting with rigorous glucose targets in many institutions. However, the same cannot be said for most non-ICU inpatient admissions. Rather, anecdotal evidence suggests that inpatient management is arbitrary and often leads to either no treatment at all or wide fluctuations in glucose when traditional management strategies are employed. In this project and according to the data set, we will use supervised machine learning techniques, particularly, classification to predict the outcome of a diabetic patient based on set of features which are thought to be very important and which are likely able to explain the variability in disease outcome in different patients.

# Approach

In this project, we are going to propose a set of questions which tailor or guide the overall data analysis approach that will be taken in order to better understand the data and gain more insights from it to instruct future decisions. There are two medically important predictions to be made which most likely are important for both diagnosis and prognosis of diabetes in different patients.

# Questions

1. Is the patient diabetic or not based on a set of important factors such as &quot;Serum Glucose level&quot;, &quot;number of diagnoses&quot;, Type of primary diagnosis and secondary diagnoses, history of diabetes related medications, insulin intake and laboratory related procedures?
2. Will the patient be readmitted to the hospital again? and If so, what is the expected relapse or readmission of the patient given his/her current representation or state of health?

# Data Description:

The data that will be collected from ([https://archive.ics.uci.edu/ml/datasets/Diabetes+130-US+hospitals+for+years+1999-2008#](https://archive.ics.uci.edu/ml/datasets/Diabetes+130-US+hospitals+for+years+1999-2008#))

# Columns

Table 1: List of features and their descriptions in the initial dataset (the dataset is also available at the website of Data Mining and Biomedical Informatics Lab at VCU ([http://www.cioslab.vcu.edu/)](http://www.cioslab.vcu.edu/)))

|     Feature name               |     Type       |     Description and   values                                                                         |     % missing    |          |
|--------------------------------|----------------|------------------------------------------------------------------------------------------------------|------------------|----------|
|     Encounter ID               |     Numeric    |     Unique identifier of   an encounter                                                              |     0%           |          |
|     Patient number             |     Numeric    |     Unique identifier of a patient                                                                   |     0%           |          |
|     Race                       |     Nominal    |     Values: Caucasian,   Asian, African American, Hispanic, and other                                |     2%           |          |
|     Gender                     |     Nominal    |     Values: male, female, and   unknown/invalid                                                      |     0%           |          |
|     Age                        |     Nominal    |     Grouped in 10-year   intervals: [0,   10), [10,   20), ..., [90, 100)                            |     0%           |          |
|     Weight                     |     Numeric    |     Weight in pounds.                                                                                |     97%          |          |
|     Admission type             |     Nominal    |     Integer identifier   corresponding to 9 distinct values, for example, emergency, urgent,         |     0%           |          |
|                                |                |     elective, newborn, and not available                                                             |                  |          |
|                                |                |                                                                                                      |                  |          |
|     Discharge   disposition    |     Nominal    |     Integer identifier corresponding to   29 distinct values, for example, discharged to             |     0%           |          |
|                                |                |     home, expired, and   not available                                                               |                  |          |
|                                |                |                                                                                                      |                  |          |
|     Admission source           |     Nominal    |     Integer identifier corresponding to   21 distinct values, for example, physician referral,       |     0%           |          |
|                                |                |     emergency room, and transfer from a   hospital                                                   |                  |          |
|                                |                |                                                                                                      |                  |          |
|     Time in hospital           |     Numeric    |     Integer number of days between   admission and discharge                                         |     0%           |          |
|     Payer code                 |     Nominal    |     Integer identifier   corresponding to 23 distinct values, for example, Blue Cross\Blue           |     52%          |          |
|                                |                |     Shield, Medicare, and self-pay                                                                   |                  |          |
|                                |                |                                                                                                      |                  |          |
|                                |                |     Integer identifier of a specialty of   the admitting physician, corresponding to 84 distinct     |                  |          |
|     Medical specialty          |     Nominal    |     values, for example,   cardiology, internal medicine, family\general practice, and               |     53%          |          |
|     Number of lab              |                |     surgeon                                                                                          |                  |          |
|                                |     Numeric    |     Number of lab tests   performed during the encounter                                             |     0%           |          |
|     procedures                 |                |                                                                                                      |                  |          |
|                                |                |                                                                                                      |                  |          |
|     Number of                  |     Numeric    |     Number of procedures (other than lab   tests) performed during the encounter                     |     0%           |          |
|     procedures                 |                |                                                                                                      |                  |          |
|                                |                |                                                                                                      |                  |          |
|     Number of                  |     Numeric    |     Number of distinct   generic names administered during the encounter                             |     0%           |          |
|     medications                |                |                                                                                                      |                  |          |
|                                |                |                                                                                                      |                  |          |
|     Number of   outpatient     |     Numeric    |     Number of outpatient visits of the   patient in the year preceding the encounter                 |     0%           |          |
|     visits                     |                |                                                                                                      |                  |          |
|                                |                |                                                                                                      |                  |          |
|     Number of                  |     Numeric    |     Number of emergency   visits of the patient in the year preceding the encounter                  |     0%           |          |
|     emergency visits           |                |                                                                                                      |                  |          |
|                                |                |                                                                                                      |                  |          |
|     Number of   inpatient      |     Numeric    |     Number of inpatient visits of the   patient in the year preceding the encounter                  |     0%           |          |
|     visits                     |                |                                                                                                      |                  |          |
|                                |                |                                                                                                      |                  |          |
|     Diagnosis 1                |     Nominal    |     The primary diagnosis   (coded as first three digits of ICD9); 848 distinct values               |     0%           |          |
|     Diagnosis 2                |     Nominal    |     Secondary diagnosis (coded as first   three digits of ICD9); 923 distinct values                 |     0%           |          |
|     Diagnosis 3                |     Nominal    |     Additional secondary   diagnosis (coded as first three digits of ICD9); 954 distinct             |     1%           |          |
|                                |                |     values                                                                                           |                  |          |
|                                |                |                                                                                                      |                  |          |
|     Number of   diagnoses      |     Numeric    |     Number of diagnoses entered to the   system                                                      |     0%           |          |
|     Glucose serum test         |     Nominal    |     Indicates the range   of the result or if the test was not taken. Values: “>200,” “>300,”        |     0%           |          |
|     result                     |                |     “normal,” and “none” if not measured                                                             |                  |          |
|                                |                |                                                                                                      |                  |          |
|                                |                |     Indicates the range of the result or   if the test was not taken. Values: “>8” if the result     |                  |          |
|     A1c test result            |     Nominal    |     was greater than 8%,   “>7”   if the result was greater than 7% but less than 8%, “normal”       |     0%           |          |
|                                |                |     if the result was less than 7%, and   “none” if not measured.                                    |                  |          |
|     Change of                  |     Nominal    |     Indicates if there   was a change in diabetic medications (either dosage or generic              |     0%           |          |
|     medications                |                |     name). Values: “change” and “no   change”                                                        |                  |          |
|                                |                |                                                                                                      |                  |          |
|     Diabetes   medications     |     Nominal    |     Indicates if there was any diabetic   medication prescribed. Values: “yes” and “no”              |     0%           |          |
|                                |                |     For the generic   names: metformin, repaglinide, nateglinide, chlorpropamide,                    |                  |          |
|                                |                |     glimepiride, acetohexamide,   glipizide, glyburide, tolbutamide, pioglitazone,                   |                  |          |
|     24 features for            |                |     rosiglitazone,   acarbose, miglitol, troglitazone, tolazamide, examide, sitagliptin, insulin,    |                  |          |
|                                |     Nominal    |     glyburide-metformin, glipizide-metformin,   glimepiride-pioglitazone,                            |     0%           |          |
|     medications                |                |     metformin-rosiglitazone,   and metformin-pioglitazone, the feature indicates whether             |                  |          |
|                                |                |                                                                                                      |                  |          |
|                                |                |     the   drug was prescribed or there was a change in the dosage. Values: “up” if the   dosage      |                  |          |
|                                |                |     was increased   during the encounter, “down” if the dosage was decreased, “steady” if the        |                  |          |
|                                |                |     dosage did not   change, and “no” if the drug was not prescribed                                 |                  |          |
|                                |                |     Days to inpatient readmission.   Values: “<30”   if the patient was readmitted in less than      |                  |          |
|     Readmitted                 |     Nominal    |     30 days, “>30” if the patient was   readmitted in more than 30 days, and “No” for no             |     0%           |          |
|                                |                |     record of readmission.                                                                           |                  |          |

# End User Benefits

1. Early prediction of diabetes in susceptible patients is very important equally for clinicians in diagnosing the disease and for patients to be guided on their treatment plan and government to be able to evaluate and predict the cost on treatments in advance and set up plans and strategies to support patients and their families during the treatment.
2. Early prediction of re-admission is very important not only to patients or clinicians but also to the hospitals themselves, to proactively respond and prepare beds and equipment to treat those patients.

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