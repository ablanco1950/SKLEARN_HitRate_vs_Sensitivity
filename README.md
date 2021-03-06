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


                                           SUSY                              HASTIE                                   ABALONE
                      SUSY_sklearn_with_test_out_of_train.py    HASTIE_sklearn_with_test_out_of_train.py        ABALONE_sklearn.py
                                  

NAIVE BAYES__________________________74.30%___________________________________74,25%____________________________________57,18%


RANDOM FOREST_______________________95,81%___________________________________100,00%_________________________________64,75%

ADABOOST_____________________78.05%___________________________________82.83%___________________________________61,87%

GRADIENT BOOST_______________________78.47____________________________________88,29%____________________________________64,44%

LOGISTIC REGRESSION__________________76.83%___________________________________51,45%____________________________________62,83%


DECISION TREE__________________100.00%_____________________________ 100.00%______________________________54,88%



TESTS WITH SUSY.CSV:

The tests showed in the table before have been got considering te first 4.500.000 records of SUSY as training file and the last 500.000 records as test file.
  
In order to be able to individually monitor each record that may appear as wrong or true according to the different records, a selection of only 20 SUSY.csv records that make up the SUSY20.txt file is made.
In this file, the classes of the first three registers are changed and saved as SUSY20bad.txt.
Both files are attached for convenience.

Changing line 21 in the SUSY_sklearn_with_test_out_train.py program to consider SUSY20.txt or SUSY 20bad.txt as the test file and removing the # comment mark on line 23, the following results are obtained:


                                        SUSY20.txt           SUSY20bad.txt              TIME
                                      hits/failures          hits/failures             seconds
                                  

NAIVE BAYES______________________________16/4___________________13/7______________________3,3
(GaussianNB)


RANDOM FOREST_________________________16/4____ ______________18/2______________________2.091,8
(RandomForestClassifier)


ADABOOST________________________________16/4__________________13/7________________________410
(AdaBoostClassifier)

GRADIENT BOOST_________________________16/4__________________13/7________________________1.522
(GradientBoostClassifier)

LOGISTIC REGRESSION____________________15/5__________________12/8________________________4.112
(LogisticRegression)

DECISION TREE___________________________20/0___________________19/1_______________________89,9
(DecisionTreeClassifier)

CONCLUSIONS:

DECISION TREE:
gives wrong results, if it gave 20 hits and 0 errors, changing the class of the first three records should have given 17 hits and 3 errors

RANDOM FOREST, GRADIENT BOOST and LOGISTIC REGRESSION:
They are impracticably slow. To check errors you would have to check each record by an expert

NAIVE BAYES and ADABOOST
The errors and successes should be verified by an expert, the results coincide with those obtained in another project that is detailed below. ADABOOST has an execution time more than 100 times higher than NAIVE BAYES

Comparing with other program and environment:

Project https://github.com/ablanco1950/SUSY_WEIGHTED_V1 results are obtained with a hit rate higher than NAIVE BAYES



TESTS WITH HASTIE:

The tests showed in the first table  have been got considering te first 3.177 records of SUSY as training file and the last 1.044 records as test file.
  
In order to be able to individually monitor each record that may appear as wrong or true according to the different records, a selection of only 20 HASTIE records that make up the Hastie10_2_20.txt file is made.

In this file, the classes of the first three registers are changed and saved as Hastie10_2_20bad.txt.
Both files are attached for convenience.

Changing line 20 in the HASTIE_sklearn_with_test_out_train.py program, taking off the comment indicator #, and also taking off the # in line 23, considering Hastie10_2_20.txt as test file or taking off the # in line 24 in wich case Hastie10_2_20bad.txt would be considered test file, the following results are obtained:


                                    Hastie10_2_20.txt    Hastie10_2_20bad.txt           TIME
                                      hits/failures          hits/failures             seconds
                                  

NAIVE BAYES______________________________15/5___________________14/6______________________3,3
(GaussianNB)


RANDOM FOREST_________________________20/0____ ______________17/3______________________2.091,8
(RandomForestClassifier)


ADABOOST________________________________16/4__________________13/7________________________410
(AdaBoostClassifier)

GRADIENT BOOST_________________________16/4__________________13/7________________________1.522
(GradientBoostClassifier)

LOGISTIC REGRESSION____________________9/11__________________10/10________________________4.112
(LogisticRegression)

DECISION TREE___________________________20/0___________________19/1_______________________89,9
(DecisionTreeClassifier)

CONCLUSIONS:

DECISION TREE:
gives wrong results, if it gave 20 hits and 0 errors, changing the class of the first three records should have given 17 hits and 3 errors

For the rest, given the diferrences in the results for each model and as are only 20 records to check, the results should be checked manually by an expert.

REFERENCES:

To the SUSY.csv file:

The results offered on the website: http://physics.bu.edu/~pankajm/ML-Notebooks/HTML/NB5_CVII-logreg_SUSY.html

"Using Pyspark Environment for Solving a Big Data Problem: Searching for Supersymmetric Particles" Mourad Azhari, Abdallah Abarda, Badia Ettaki, Jamal Zerouaoui, Mohamed Dakkon International Journal of Innovative Technology and Exploring Engineering (IJITEE) ISSN: 2278-3075, Volume-9 Issue-7, May 2020 https://www.researchgate.net/publication/341301008_Using_Pyspark_Environment_for_Solving_a_Big_Data_Problem_Searching_for_Supersymmetric_Particles

The one already referenced before: https://github.com/ablanco1950/SUSY_WEIGHTED_V1 (78,09% hit rate)


To the HASTIE file:

Implementation of AdaBoost classifier

https://github.com/jaimeps/adaboost-implementation

https://github.com/ablanco1950/HASTIE_NAIVEBAYES (87,13% hit rate)
https://github.com/ablanco1950/HASTIE_Corrected_HitRate_vs_Sensitivity
https://github.com/ablanco1950/HASTIE_Corrected_DecisionTree (99% hit rate)

To the ABALONE file:

https://archive.ics.uci.edu/ml/datasets/abalone, especially the download of the Data Set Description link, you can check the low hit rates
achieved with this file.

https://github.com/ablanco1950/ABALONE_NAIVEBAYES_WEIGHTED_ADABOOST (58% hit rate)
https://github.com/ablanco1950/ABALONE_DECISIONTREE_C4-5 (58% hit rate)
