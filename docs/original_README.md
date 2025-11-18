# College-Project
this is my college project on prediction of cardiac arrest using machine learning



# Heart attack Prediction using Machine Learning

In this notebook we are going to perform Exploratory Data Analysis and use various Machine Learning Models to predict whether the patient has heart disease or not depending on the values of various features. We will be using Bokeh and a little bit of Seaborn to plot the graphs.


#                               About the Datasets

This notebook contains 4 databases concerning heart disease diagnosis. All attributes are    numeric-valued. The data was collected from the four following locations:

                1. Cleveland Clinic Foundation (cleveland.csv)
                2. Hungarian Institute of Cardiology, Budapest (hungarian.csv)
                3. V.A. Medical Center, Long Beach, CA (long-beach-va.csv)
                4. University Hospital, Zurich, Switzerland (switzerland.csv)


Each database has the same instance format. While the databases have 76 raw attributes, only 14 of them are actually used. This database contains 76 attributes, but all published experiments refer to using a subset of 14 of them. In particular, the Cleveland database is the only one that has been used by ML researchers to this date. The "goal" field refers to the presence of heart disease in the patient. It is integer valued from 0 (no presence) to 4.
Experiments with the Cleveland database have concentrated on simply attempting to distinguish presence (values 1,2,3,4) from absence (value 0).  


Number of Instances: 
             
              Database:               Number of instances:

              Cleveland:                      303
              Hungarian:                      294
              Switzerland:                    123
              Long Beach VA:                  200


Number of Attributes: 76 (including the predicted attribute)


The authors of the databases have requested that any publications resulting from the use of the data include the names of the principal investigator responsible for the data collection at each institution. They would be:

        a. Hungarian Institute of Cardiology. Budapest: Andras Janosi, M.D.
        b. University Hospital, Zurich, Switzerland: William Steinbrunn, M.D.
        c. University Hospital, Basel, Switzerland: Matthias Pfisterer, M.D.
        d. V.A. Medical Center, Long Beach and Cleveland Clinic Foundation:
	     Robert Detrano, M.D., Ph.D.

The dataset consists of 303 individual data. There are 14 columns in the dataset, which      are described below:

        1. Age: displays the age of the individual.

        2. Sex: displays the gender of the individual using the following format:
                    1 = male
                    0 = female

        3. Cp (Chest-pain type): displays the type of chest-pain experienced by the
           individual using the following format:
                    1 = typical angina
                    2 = atypical angina
                    3 = non-anginal pain
                    4 = asymptotic

        4. TrestBPS (Resting Blood Pressure): Displays the resting blood pressure value of
           an individual in mmHg (unit). It can take continuous values from 94 to 200.

        5. Chol (Serum Cholesterol): Displays the serum cholesterol in mg/dl (unit)

        6. Fbs (Fasting Blood Sugar): Compares the fasting blood sugar value of an
           individual with 120mg/dl:
                    1 (true) = Fasting blood sugar > 120mg/dl 
                    0 (False) = Fasting blood sugar < 120mg/dl 

        7. RestECG (Resting ECG): displays resting electrocardiographic results
                    0 = normal
                    1 = having ST-T wave abnormality
                    2 = left ventricular hypertrophy

        8. Thalach (Max heart rate achieved): displays the max heart rate achieved by an
           individual. It can take continuous value from 71 to 202.

        9. Exang (Exercise induced angina): Angina is a type of chest pain caused by reduced
           blood flow to the heart.
                    1 = yes
                    0 = no

        10.	OldPeak (ST depression induced by exercise relative to rest): displays the value
            which is an integer or float.

        11.	Slope (Peak exercise ST segment):
                    1 = upsloping
                    2 = flat
                    3 = down sloping

        12.	Ca (Number of major vessels (0–3) colored by fluoroscopy): displays the value as
            integer or float.

        13.	Thal: displays the thalassemia:
                    3 = normal
                    6 = fixed defect
                    7 = reversible defect

        14.	Target (Diagnosis of heart disease): Displays whether the individual is
            suffering from heart disease or not:
                    0 = absence
                    1, 2, 3, 4 = present.



# Importing Pickled files

Pickle in Python is primarily used in serializing and deserializing a Python object structure. In other words, it’s the process of converting a Python object into a byte stream to store it in a file/database, maintain program state across sessions, or transport data over the network. The pickled byte stream can be used to re-create the original object hierarchy by unpickling the stream. This whole process is similar to object serialization in Java or .Net.

When a byte stream is unpickled, the pickle module creates an instance of the original object first and then populates the instance with the correct data. To achieve this, the byte stream contains only the data specific to the original object instance. But having just the data alone may not be sufficient. To successfully unpickle the object, the pickled byte stream contains instructions to the unpickler to reconstruct the original object structure along with instruction operands, which help in populating the object structure.

According to the pickle module documentation, the following types can be pickled:

    1. None, true, and false
    2. Integers, long integers, floating point numbers, complex numbers
    3. Normal and Unicode strings
    4. Tuples, lists, sets, and dictionaries containing only picklable objects
    5. Functions defined at the top level of a module
    6. Built-in functions defined at the top level of a module
    7. Classes that are defined at the top level of a module
