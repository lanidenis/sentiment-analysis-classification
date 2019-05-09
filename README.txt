README.txt

Jelani Denis, Jaeyoon Jung
Assignment 1
COS 424 

Our code is located inside of a .ipynb file which can be opened with Jupyter Notebook.  The installation for jupyter notebook follows these commands: (http://jupyter.org/install)

	python3 -m pip install --upgrade pip
	python3 -m pip install jupyter

We use python3 for our implementation. 

Once opened inside jupyter, there is really nothing you need to run.
We have run all the pertinent code and kept the output visible for you.

The first half of the document consists only of import statements and function definitions. 

The second half shows the printed results of classification runs over a number of different classifiers, feature selection, and feature extraction combinations.  All runs are performed by a call to the "execute()" function.

Near the end of the document, we show results for a 5-fold cross-validation on our two top classifiers.

Near the end, we also show the results for a function which determines the most informative features for both classifiers.

At the very end, is the function "extend()". It adds the 1.5 million samples of Twitter data to the train and test sets.  You should not call this function.  We used to to learn a Multinomial NB classifier and have reported the results in our write up. 

A description of all functions is below:

	Function "load_data" loads the data from train.txt and test.txt.  Make sure they are in the same directory as this .ipynb file.

	Function "search(sequence)" you can play with if you like. Given an array of search strings, it outputs the number of occurences of each one for both positive and negative samples.

	Function "preprocess(raw)" is used by sci-kit learn classifiers later on.

	Function "maximumN(mydict,N)" computes the N most popular words in both the positive and negative class subsets.

	Functions "new_features()", "warp()", and "add_features()" are for internal use.  The work together to add different features like sample length to the bag-of-words representations.

	Function "feature_extension()" calls the above functions.

	Function "count()" is for internal use.  It creates bag-of-words feature vectors based on certain parameter specifications.

	Function "execute()" performs classification based on the arguments you pass to it controlling the type of classifier, word-vector-type, feature selection, and feautre addition if any.


