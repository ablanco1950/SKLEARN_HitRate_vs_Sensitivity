Using the Sklearn classifiers: Naive Bayes, Random Forest, Adaboost and Gradient Boost, good success rates are observed in a very simple manner. In this work sensitivity is also considered.

Requirements:

Spyder 4

In drive C: the files with which the tests have been carried out should be found:

SUSY.csv (download from https://archive.ics.uci.edu/ml/datasets/SUSY)

HASTIE: attached file Hastie10_2.csv, obtained following the instructions in
 https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_hastie_10_2

ABALONE: to download from (https://archive.ics.uci.edu/ml/datasets/abalone))

By executing the attached programs indicated, the following percentages of successes are obtained in the test:


                                           SUSY                   HASTIE                  ABALONE
                                      SUSY_sklearn.py          HASTIE_sklearn.py       ABALONE_sklearn.py

NAIVE BAYES                               74.41%                  75.04%                    54.9%
(GaussianNB)


RANDOM FOREST                             78,1%                    81,75%                    65,07%
(RandomForestClassifier)


ADABOOST                                   77.6%                   80.75%                   61.72%
(AdaBoostClassifier)

GRADIENT BOOST                             78.26%                  84,8%                    64.7%
(GradientBoostClassifier)

  
Then, when repeating the test for SUSY with the attached program SUSY_sklearn_with_test_out_train.py in which the test file is separated from the training file and
it consists of the attached Susy20.txt file, which must be transferred to the C: drive, which contains only 20 records extracted from SUSY.csv, and executing SUSY_sklearn.py from spyder.
It will be seen that all classifiers  shows 13 hits and 7 failures except RANDON FOREST  wich shows 17 hits and 3 failures.

If, intentionally, the class of the first 3 records of Susy20.txt is changed, or the attached Susy20Bad.txt file is used, which is the Susy20.txt with the three errors introduced, in which case line 16 of the SUSY_sklearn.py program would have to be changed, changing the file assignment. It is observed that classifiers reflects the 3 misses added except RANDOM FOREST that shows 18 errors and 2 hits, reflecting only one of the three errors introduced.

Due to memory problems only 30000 records of SUSY are considered.

Comparing with other program and environment:

Downloading and installing the procedure found at https://github.com/ablanco1950/SUSY_WEIGHTED_V1 and running AssignClassWithSusyWeighted_v1.bat,
changing before the only line of the procedure, so that it is:

java -jar AssignClassWithSusyWeighted_v1.jar c: \ SusyWeighted78PercentHits.txt c: \ SUSY20.txt 0.0 4500000.0 0.0 4500000.0 5000000.0

So change the reference to SUSY.csv to SUSY20.txt, you get 16 hits and 4 misses.

 By repeating the procedure referencing the SUSY20Bad.txt file instead of SUSY20.txt, that is, referencing the SUSY20.txt to which the classes of
the first three records has been changed, and executing AssignClassWithSusyWeighted_v1.bat, first changing the only line of the procedure, so that it is:

java -jar AssignClassWithSusyWeighted_v1.jar c:\SusyWeighted78PercentHits.txt c:\SUSY20Bad.txt 0.0 4500000.0 0.0 4500000.0 5000000.0

13 hits and 7 failures are obtained, that is, the 3 errors introduced when changing the classes of the first 3 registers have been detected.

REFERENCES:

To the SUSY.csv file:

The results offered on the website: http://physics.bu.edu/~pankajm/ML-Notebooks/HTML/NB5_CVII-logreg_SUSY.html

"Using Pyspark Environment for Solving a Big Data Problem: Searching for Supersymmetric Particles" Mourad Azhari, Abdallah Abarda, Badia Ettaki, Jamal Zerouaoui, Mohamed Dakkon International Journal of Innovative Technology and Exploring Engineering (IJITEE) ISSN: 2278-3075, Volume-9 Issue-7, May 2020 https://www.researchgate.net/publication/341301008_Using_Pyspark_Environment_for_Solving_a_Big_Data_Problem_Searching_for_Supersymmetric_Particles

The one already referenced before: https: //github.com/ablanco1950/SUSY_WEIGHTED_V1

To the HASTIE file:

Implementation of AdaBoost classifier

https://github.com/jaimeps/adaboost-implementation

https://github.com/ablanco1950/HASTIE_NAIVEBAYES

To the ABALONE file:

https://archive.ics.uci.edu/ml/datasets/abalone, especially the download of the Data Set Description link, you can check the low hit rates
achieved with this file, however with GRADIENT FOREST (RandomForestClassifier) ​​100% hit rates are achieved not only considering the 3 classes in
that the 29 classes are summarized, to consider the 29 classes ( run in spyder the attached ABALONE_with_29Classes_sklearn.py)

https://github.com/ablanco1950/ABALONE_NAIVEBAYES_WEIGHTED_ADABOOST
https://github.com/ablanco1950/ABALONE_DECISIONTREE_C4-5
