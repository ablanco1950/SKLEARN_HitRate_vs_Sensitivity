Using the Sklearn classifiers: Naive Bayes, Random Forest, Adaboost, Gradient Boost, Logistic Regression and Decision Tree good success rates are observed in a very simple manner. In this work sensitivity is also considered.
Treating each record individually, differences are found in the results for each record depending on the model used, which is hidden in treatments that compute a total volume of data

Requirements:

Spyder 4

In drive C: the files with which the tests have been carried out should be found:

SUSY.csv (download from https://archive.ics.uci.edu/ml/datasets/SUSY)

HASTIE: attached file Hastie10_2.csv, obtained following the instructions in
 https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_hastie_10_2

ABALONE: to download from (https://archive.ics.uci.edu/ml/datasets/abalone))

By executing the attached programs indicated, the following percentages of successes are obtained in the test:


                                           SUSY                   HASTIE                  ABALONE
                      SUSY_sklearn_with_test_out_of_train.py    HASTIE_sklearn.py       ABALONE_sklearn.py
                                  

NAIVE BAYES______________________________74.30%___________________75.04%______________________54.9%
(GaussianNB)


RANDOM FOREST_________________________95,81%____ ______________81,75%______________________65,07%
(RandomForestClassifier)


ADABOOST________________________________78.05%____________________80.75%______________________61.72%
(AdaBoostClassifier)

GRADIENT BOOST_________________________78.47%___________________-----_______________________-----
(GradientBoostClassifier)

LOGISTIC REGRESSION____________________76.83%___________________-----_______________________-----
(LogisticRegression)

DECISION TREE__________________________100.00%___________________--- --_______________________-----
(DecisionTreeClassifier)



TESTS WITH SUSY.CSV:

The tests showed in the table before have been got considering te first 4.500.000 records of SUSY as training file and the last 500.000 records as test file.
  
Then, when repeating the test for SUSY with the attached program SUSY_sklearn_with_test_out_train.py in which the test file is separated from the training file and
it consists of the attached Susy20.txt file, which must be transferred to the C: drive, which contains only 20 records extracted from SUSY.csv, and executing SUSY_sklearn_with_test_out_train.py from spyder.
It will be seen that all classifiers  shows 16 hits and 4 failures except RANDON FOREST  wich shows 18 hits and 2 failures.

If, intentionally, the class of the first 3 records of Susy20.txt is changed, or the attached Susy20Bad.txt file is used, which is the Susy20.txt with the three errors introduced, in which case line 16 of the SUSY_sklearn.py program would have to be changed, changing the file assignment. It is observed that classifiers reflects the 3 misses added, 13 hits 7 misses, except RANDOM FOREST that shows 17 hits and 3 misses, reflecting only one of the three errors introduced.

Due to memory problems only 300.000 records of SUSY are considered.

Comparing with other program and environment:

TESTS with SUSY.csv:

The tests showed in the previous table have been developed considering the first 4.500.000 records as training file and the last 500.000 records as test file 

Downloading and installing the procedure found at https://github.com/ablanco1950/SUSY_WEIGHTED_V1 and running AssignClassWithSusyWeighted_v1.bat,
changing before the only line of the procedure, so that it is:

java -jar AssignClassWithSusyWeighted_v1.jar c:\SusyWeighted78PercentHits.txt c:\SUSY20.txt 0.0 4500000.0 0.0 4500000.0 5000000.0

So change the reference to SUSY.csv to SUSY20.txt, you get 16 hits and 4 misses.

 By repeating the procedure referencing the SUSY20Bad.txt file instead of SUSY20.txt, that is, referencing the SUSY20.txt to which the classes of
the first three records has been changed, and executing AssignClassWithSusyWeighted_v1.bat, first changing the only line of the procedure, so that it is:

java -jar AssignClassWithSusyWeighted_v1.jar c:\SusyWeighted78PercentHits.txt c:\SUSY20Bad.txt 0.0 4500000.0 0.0 4500000.0 5000000.0

13 hits and 7 failures are obtained, that is, the 3 errors introduced when changing the classes of the first 3 registers have been detected.

REFERENCES:

To the SUSY.csv file:

The results offered on the website: http://physics.bu.edu/~pankajm/ML-Notebooks/HTML/NB5_CVII-logreg_SUSY.html

"Using Pyspark Environment for Solving a Big Data Problem: Searching for Supersymmetric Particles" Mourad Azhari, Abdallah Abarda, Badia Ettaki, Jamal Zerouaoui, Mohamed Dakkon International Journal of Innovative Technology and Exploring Engineering (IJITEE) ISSN: 2278-3075, Volume-9 Issue-7, May 2020 https://www.researchgate.net/publication/341301008_Using_Pyspark_Environment_for_Solving_a_Big_Data_Problem_Searching_for_Supersymmetric_Particles

The one already referenced before: https://github.com/ablanco1950/SUSY_WEIGHTED_V1 (78,09% hit rate)


To the HASTIE file:

Implementation of AdaBoost classifier

https://github.com/jaimeps/adaboost-implementation

https://github.com/ablanco1950/HASTIE_NAIVEBAYES (87,13% hit rate)

To the ABALONE file:

https://archive.ics.uci.edu/ml/datasets/abalone, especially the download of the Data Set Description link, you can check the low hit rates
achieved with this file.

https://github.com/ablanco1950/ABALONE_NAIVEBAYES_WEIGHTED_ADABOOST (58% hit rate)
https://github.com/ablanco1950/ABALONE_DECISIONTREE_C4-5 (58% hit rate)
