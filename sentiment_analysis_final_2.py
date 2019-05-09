
# coding: utf-8

# In[2]:


import sklearn
import numpy as np
import scipy.sparse
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn import svm
import time
# read in dataset
train_path = "train.txt"
test_path = "test.txt"


# In[3]:


def load_data(path):
    with open(path, "r") as f:
        lines = f.readlines()
    
    classes = []
    #samples = []
    docs = []
        
    for line in lines:
        classes.append(int(line.rsplit()[-1]))
        #samples.append(line.rsplit()[0])
        #raw = line.decode('latin1')
        raw = ' '.join(line.rsplit()[1:-1])
        docs.append(raw)
    
    return (docs, classes)


# In[4]:


X_train, Y_train = load_data(train_path)
X_test, Y_test = load_data(test_path)

def search(sequence):
    result = []
    for word in sequence:   
        counter = [1 if word in x else 0 for x in X_train]
        indexes = []
        for i in range(len(counter)):
            if counter[i] != 0:
                indexes.append(i)
        positive = 0
        negative = 0
        for i in range(len(indexes)):
            if Y_train[indexes[i]] == 1:
                positive += 1
            else:
                negative += 1
        result.append((word,positive,negative))
    return result

chars = ['{','}','#','%','&','\(','\)','\[','\]','<','>',',', '!', '.', ';', 
'?', '*', '\\', '\/', '~', '_','|','=','+','^',':','\"','\'','@','-']
for element in search(["!", "'", "?"]):
    print(element)


# In[5]:


from sklearn.feature_extraction.text import CountVectorizer
import nltk, re
from nltk.corpus import stopwords

porter = nltk.PorterStemmer() # also lancaster stemmer
wnl = nltk.WordNetLemmatizer()
stopWords = stopwords.words("english")
chars = ['{','}','#','%','&','\(','\)','\[','\]','<','>',',', '!', '.', ';', 
'?', '*', '\\', '\/', '~', '_','|','=','+','^',':','\"','\'','@','-']

def preprocess(raw):
    line = re.sub('[%s]' % ''.join(chars), ' ', (raw))
    words = line.split(' ')
    words = [w.lower() for w in words]
    words = [w for w in words if w not in stopWords]
    words = [wnl.lemmatize(w) for w in words]
    processed = ' '.join([porter.stem(w) for w in words])
    
    return processed
    


# In[12]:


#Finding most common words in pos/neg classes
import heapq

positives = {}
negatives = {}

for i, sentence in enumerate(X_train):
    words = sentence.split(" ")
    #add bigrams
    #for i in range(len(words)-1):
    #    words.append(words[i]+" "+words[i+1])
    if Y_train[i] == 1:
        for word in words:
            if word in stopWords:
                continue
            if word in positives:
                positives[word] += 1  
            else:
                positives[word] = 1
    else:
        for word in words:
            if word in stopWords:
                continue
            if word in negatives:
                negatives[word] += 1
            else:
                negatives[word] = 1 

def maximumN(mydict, N):
    myheap = []
    count = 0
    for key in mydict:
        if count == N:
            heapq.heappush(myheap, (mydict[key], key))
            heapq.heappop(myheap)
            
        else:
            heapq.heappush(myheap, (mydict[key], key))
            count += 1
            
    print(myheap)

stopWords.extend(["I","movie","It","The","This"])
maximumN(positives, 15)
maximumN(negatives, 15)


# In[7]:


def new_features(char, word, cap):
    toReturn = []
    if char or word or cap:
        print("adding new features")
    if char: #compute character length feature
        char_train = [len(x) for x in X_train]
        char_test = [len(x) for x in X_test]
        toReturn.append((char_train, char_test))
    else:
        toReturn.append(None)

    if word: #compute word length feature
        word_train = [len(x.split(" ")) for x in X_train]
        word_test = [len(x.split(" ")) for x in X_test]
        toReturn.append((word_train, word_test))
    else:
        toReturn.append(None)
        
    if cap: #compute cap count feature
        cap_train = [sum([1 if x.isalpha() and x.isupper() else 0 for x in row]) for row in X_train]
        cap_test = [sum([1 if x.isalpha() and x.isupper() else 0 for x in row]) for row in X_test]
        toReturn.append((cap_train, cap_test))
    else:
        toReturn.append(None)
        
    return toReturn

#print("Before")
#print("mean: ", np.mean(cap_train))
#print("sd: ", np.std(cap_train))
#print("min: ", min(cap_train))
#print("max: ", max(cap_train))

def warp(features, log, std, reg): #warp to sd of 1
    toReturn = []
    
    for i in range(0,len(features)):
        if features[i] is not None:
            if std:
                sd = np.std(features[i][0])
                train = [x / sd for x in features[i][0]]
                test = [x / sd for x in features[i][1]]
            elif log:
                train = [np.log(x) for x in features[i][0]]
                test = [np.log(x) for x in features[i][1]]
            else:
                train = features[i][0]
                test = features[i][1]
            toReturn.append((train,test))
        else:
            toReturn.append(None)
        
    return toReturn

def add_features(features, train, test):
    for i in range(0, len(features)):
        if features[i] is not None: 
            train = scipy.sparse.hstack((train,np.array(features[i][0])[:,None]))
            test = scipy.sparse.hstack((test,np.array(features[i][1])[:,None]))
    return (train,test)

#print("After")
#print("mean: ", np.mean(X_cap_train))
#print("sd: ", np.std(X_cap_train))
#print("min: ", min(X_cap_train))
#print("max: ", max(X_cap_train))


# In[8]:


import sys
#print("length of train: ", len(X_train))
#print("length of test: ", len(X_test))

features = new_features(True, True, True)
title = ["Char", "Word", "Cap"]


for i in range(len(features)):
    train = features[i][0]
    test = features[i][1]
    both = list(train)
    both.extend(test)
    print("Statistics for " + title[i] + " Count")
    print("mean: ", np.mean(both), np.mean(train), np.mean(test))
    print("sd:   ", np.std(both), np.std(train), np.std(test))
    print("min:  ", min(both), min(train), min(test))
    print("max:  ", max(both), max(train), max(test))


# In[9]:


X_train_out = []
X_test_out = []

def feature_extension(char, word, cap):
    global X_train_out, X_test_out
    #print(X_train_out.getnnz(1)) #nonzeros across row

    #add features
    features = new_features(char, word, cap) #char, word, cap
    warped = warp(features, False, True, False) #log, std, reg
    result = add_features(warped, X_train_out, X_test_out)
    X_train_out = result[0]
    X_test_out = result[1]

    #print(X_train_out.getnnz(1)) #nonzeros across row


# In[8]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn import metrics

def count(multi, maxgram, tfidf, svd, chi, char, word, cap):
    global X_train_out, X_test_out
    if tfidf:
        print("extracting tfidf...")
        counter = TfidfVectorizer(use_idf=True, preprocessor=preprocess, ngram_range=(1, maxgram))
        X_train_out = counter.fit_transform(X_train)
        X_test_out = counter.transform(X_test)
    else:
        if multi:
            print("multinomial distribution")
        else:
            print("bernoulli distribution")
        print("maxgram: " + str(maxgram))
        counter = CountVectorizer(preprocessor=preprocess, binary=multi, ngram_range=(1, maxgram))
        X_train_out = counter.fit_transform(X_train)
        X_test_out = counter.transform(X_test)
    
    if svd:
        print("SVD dimenstionality reduction")
        svd = TruncatedSVD(n_components=100)
        X_train_out = svd.fit_transform(X_train_out)
        X_test_out = svd.transform(X_test_out)
    
    elif chi:
        print("chi2 feature reduction")
        kbest = SelectKBest(chi2, k=100)
        X_train_out = kbest.fit_transform(X_train_out, Y_train)
        X_test_out = kbest.transform(X_test_out)
    feature_extension(char, word, cap)


# In[9]:


def gridsearchSVM(train_feature, train_label, test_feature, test_label):
    params = {"kernel":[ "linear", "poly", "rbf", "sigmoid"], 
           "C":[0.001, 0.01, 0.1, 1, 10, 100, 1000]}
    scoring = metrics.make_scorer(metrics.accuracy_score)
    grid = GridSearchCV(svm.SVC(random_state=0), params, cv=5,
                        scoring=scoring)
    grid.fit(train_feature, train_label)

    print("best parameters: ")
    print(grid.best_estimator_)
    preds = grid.predict(test_feature)

    print("Accuracy: " + str(metrics.accuracy_score(preds, test_label)))
    print("F1: " + str(metrics.f1_score(preds, test_label)))


# In[10]:


from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

def execute(bern, mult, rf, knn, dt, svm, multi, maxgram, tfidf, svd, chi, char, word, cap):
    global X_train_out, X_test_out
    
    start = time.time()
    if bern:
        clf = BernoulliNB()
        print("<Bernoulli NB>")
    elif mult:
        print("<Multinomial NB>")
        clf = MultinomialNB()
    elif rf:
        print("<Random Forest>")
        clf = RandomForestClassifier(n_estimators=100, random_state=0)
    elif knn:
        print("<KNN>")
        clf = KNeighborsClassifier(n_neighbors=50)
    elif dt:
        print("<Decision Tree>")
        clf = DecisionTreeClassifier(random_state=0)
    else:
        x = 1
    
    count(multi, maxgram, tfidf, svd, chi, char, word, cap)
    
    if svm:
        print("<SVM>")
        gridsearchSVM(X_train_out, Y_train, X_test_out, Y_test)
    else:
        clf.fit(X_train_out, Y_train)
        Y_pred = clf.predict(X_test_out)
        print("Accuracy: " + str(metrics.accuracy_score(Y_pred, Y_test)))
        print("F1: " + str(metrics.f1_score(Y_pred, Y_test)))  
        
    end = time.time()
    print("run time: " + str(end - start))
    


# In[11]:


execute(bern=True, mult=False, rf=False, knn=False, dt=False, svm=False, 
        multi=False, maxgram=1, tfidf=False, svd=False, chi=False, 
        char=False, word=False, cap=False)


# In[12]:


execute(bern=False, mult=True, rf=False, knn=False, dt=False, svm=False, 
        multi=True, maxgram=1, tfidf=False, svd=False, chi=False, 
        char=False, word=False, cap=False)


# In[13]:


execute(bern=False, mult=False, rf=False, knn=True, dt=False, svm=False, 
        multi=True, maxgram=1, tfidf=False, svd=False, chi=False, 
        char=False, word=False, cap=False)


# In[14]:


execute(bern=False, mult=False, rf=False, knn=False, dt=True, svm=False, 
        multi=True, maxgram=1, tfidf=False, svd=False, chi=False, 
        char=False, word=False, cap=False)


# In[15]:


execute(bern=False, mult=False, rf=True, knn=False, dt=False, svm=False, 
        multi=True, maxgram=1, tfidf=False, svd=False, chi=False, 
        char=False, word=False, cap=False)


# In[16]:


execute(bern=False, mult=False, rf=False, knn=False, dt=False, svm=True, 
        multi=True, maxgram=1, tfidf=False, svd=False, chi=False, 
        char=False, word=False, cap=False)


# In[17]:


execute(bern=True, mult=False, rf=False, knn=False, dt=False, svm=False, 
        multi=False, maxgram=2, tfidf=False, svd=False, chi=False, 
        char=False, word=False, cap=False)


# In[18]:


execute(bern=False, mult=True, rf=False, knn=False, dt=False, svm=False, 
        multi=True, maxgram=2, tfidf=False, svd=False, chi=False, 
        char=False, word=False, cap=False)


# In[19]:


execute(bern=False, mult=False, rf=False, knn=False, dt=True, svm=False, 
        multi=True, maxgram=2, tfidf=False, svd=False, chi=False, 
        char=False, word=False, cap=False)


# In[20]:


execute(bern=False, mult=False, rf=True, knn=False, dt=False, svm=False, 
        multi=True, maxgram=2, tfidf=False, svd=False, chi=False, 
        char=False, word=False, cap=False)


# In[21]:


execute(bern=False, mult=False, rf=False, knn=False, dt=False, svm=True, 
        multi=True, maxgram=2, tfidf=False, svd=False, chi=False, 
        char=False, word=False, cap=False)


# In[22]:


#Good
execute(bern=False, mult=True, rf=False, knn=False, dt=False, svm=False, 
        multi=True, maxgram=2, tfidf=True, svd=False, chi=False, 
        char=False, word=False, cap=False)


# In[23]:


execute(bern=False, mult=False, rf=True, knn=False, dt=False, svm=False, 
        multi=True, maxgram=2, tfidf=True, svd=False, chi=False, 
        char=False, word=False, cap=False)


# In[24]:


#good
execute(bern=False, mult=False, rf=False, knn=False, dt=False, svm=True, 
        multi=True, maxgram=2, tfidf=True, svd=False, chi=False, 
        char=False, word=False, cap=False)


# In[25]:


execute(bern=False, mult=True, rf=False, knn=False, dt=False, svm=False, 
        multi=True, maxgram=2, tfidf=True, svd=False, chi=False, 
        char=True, word=True, cap=True)


# In[26]:


execute(bern=False, mult=True, rf=False, knn=False, dt=False, svm=False, 
        multi=True, maxgram=2, tfidf=False, svd=False, chi=False, 
        char=True, word=True, cap=True)


# In[27]:


execute(bern=False, mult=False, rf=True, knn=False, dt=False, svm=False, 
        multi=True, maxgram=2, tfidf=False, svd=False, chi=False, 
        char=True, word=True, cap=True)


# In[28]:


execute(bern=False, mult=False, rf=True, knn=False, dt=False, svm=False, 
        multi=True, maxgram=2, tfidf=True, svd=False, chi=False, 
        char=True, word=True, cap=True)


# In[29]:


execute(bern=False, mult=False, rf=False, knn=False, dt=False, svm=True, 
        multi=True, maxgram=2, tfidf=False, svd=False, chi=False, 
        char=True, word=True, cap=True)


# In[30]:


#good
execute(bern=False, mult=False, rf=False, knn=False, dt=False, svm=True, 
        multi=True, maxgram=2, tfidf=True, svd=False, chi=False, 
        char=True, word=True, cap=True)


# In[31]:


execute(bern=False, mult=False, rf=True, knn=False, dt=False, svm=False, 
        multi=True, maxgram=2, tfidf=True, svd=True, chi=False, 
        char=False, word=False, cap=False)


# In[32]:


execute(bern=False, mult=False, rf=False, knn=False, dt=False, svm=True, 
        multi=True, maxgram=2, tfidf=True, svd=True, chi=False, 
        char=False, word=False, cap=False)


# In[33]:


execute(bern=False, mult=False, rf=True, knn=False, dt=False, svm=False, 
        multi=True, maxgram=2, tfidf=True, svd=False, chi=True, 
        char=False, word=False, cap=False)


# In[34]:


execute(bern=False, mult=False, rf=True, knn=False, dt=False, svm=False, 
        multi=True, maxgram=2, tfidf=True, svd=False, chi=True, 
        char=True, word=True, cap=True)


# In[35]:


execute(bern=False, mult=False, rf=False, knn=False, dt=False, svm=True, 
        multi=True, maxgram=2, tfidf=True, svd=False, chi=True, 
        char=False, word=False, cap=False)


# In[36]:


execute(bern=False, mult=False, rf=False, knn=False, dt=False, svm=True, 
        multi=True, maxgram=2, tfidf=True, svd=False, chi=True, 
        char=True, word=True, cap=True)


# In[38]:


execute(bern=False, mult=True, rf=False, knn=False, dt=False, svm=False, 
        multi=True, maxgram=2, tfidf=True, svd=False, chi=True, 
        char=False, word=False, cap=False)


# In[37]:


execute(bern=False, mult=True, rf=False, knn=False, dt=False, svm=False, 
        multi=True, maxgram=2, tfidf=True, svd=False, chi=True, 
        char=True, word=True, cap=True)


# In[39]:


X = X_train + X_test
Y = Y_train + Y_test


# In[40]:


from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import KFold, train_test_split

def validate_model(clf, features, label, char_feat):
    accuracy_result = []
    f1_result = []
    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    i = 0
    for train_ind, test_ind in kf.split(features):
        train_feat = features[train_ind]
        test_feat = features[test_ind]
        
        counter = TfidfVectorizer(use_idf=True, preprocessor=preprocess, ngram_range=(1, 2))
        train_feat_out = counter.fit_transform(train_feat)
        test_feat_out = counter.transform(test_feat)
        
        if char_feat:
            char_train = [len(x) for x in train_feat]
            char_test = [len(x) for x in test_feat]
            char_sd = np.std(char_train)
            char_train = [x / char_sd for x in char_train]
            char_test = [x / char_sd for x in char_test]
            train_feat_out = scipy.sparse.hstack((train_feat_out,np.array(char_train)[:,None]))
            test_feat_out = scipy.sparse.hstack((test_feat_out,np.array(char_test)[:,None]))
            
            word_train = [len(x.split(" ")) for x in train_feat]
            word_test = [len(x.split(" ")) for x in test_feat]
            word_sd = np.std(word_train)
            word_train = [x / word_sd for x in word_train]
            word_test = [x / word_sd for x in word_test]
            train_feat_out = scipy.sparse.hstack((train_feat_out,np.array(word_train)[:,None]))
            test_feat_out = scipy.sparse.hstack((test_feat_out,np.array(word_test)[:,None]))
            
            cap_train = [sum([1 if x.isalpha() and x.isupper() else 0 for x in row]) for row in train_feat]
            cap_test = [sum([1 if x.isalpha() and x.isupper() else 0 for x in row]) for row in test_feat]
            cap_sd = np.std(cap_train)
            cap_train = [x / cap_sd for x in cap_train]
            cap_test = [x / cap_sd for x in cap_test]
            train_feat_out = scipy.sparse.hstack((train_feat_out,np.array(cap_train)[:,None]))
            test_feat_out = scipy.sparse.hstack((test_feat_out,np.array(cap_test)[:,None]))
            
            
            
        i = i + 1
        print ('Fold {}'.format(i))
        clf.fit(train_feat_out, label[train_ind])
        Y_pred = clf.predict(test_feat_out)
        accuracy = metrics.accuracy_score(Y_pred, label[test_ind])
        f1score = metrics.f1_score(Y_pred, label[test_ind])
        accuracy_result.append(accuracy)
        f1_result.append(f1score)

        print ('Accuracy:{}'.format(accuracy))
        print ('F1 Score: {}'.format(f1score))

    print ('Overall Accuracy: {}'.format(np.mean(accuracy_result)))
    print ('Overall F1 Score: {}'.format(np.mean(f1_result)))


# In[42]:


X = X_train + X_test
Y = Y_train + Y_test
validate_model(MultinomialNB(), np.asarray(X), np.asarray(Y),False)


# In[43]:


validate_model(svm.SVC(random_state=0, C=1, kernel="linear"), np.asarray(X), np.asarray(Y),False)


# In[44]:


validate_model(svm.SVC(random_state=0, C=1, kernel="linear"), np.asarray(X), np.asarray(Y),True)


# In[62]:


def most_informative(clf, train_feat, train_label, char_feat):
    accuracy_result = []
    f1_result = []
        
    counter = TfidfVectorizer(use_idf=True, preprocessor=preprocess, ngram_range=(1, 2))
    train_feat_out = counter.fit_transform(train_feat)
    feat_name = counter.get_feature_names()
        
    if char_feat:
        char_train = [len(x) for x in train_feat]
        char_sd = np.std(char_train)
        char_train = [x / char_sd for x in char_train]
        train_feat_out = scipy.sparse.hstack((train_feat_out,np.array(char_train)[:,None]))

        word_train = [len(x.split(" ")) for x in train_feat]
        word_sd = np.std(word_train)
        word_train = [x / word_sd for x in word_train]
        train_feat_out = scipy.sparse.hstack((train_feat_out,np.array(word_train)[:,None]))

        cap_train = [sum([1 if x.isalpha() and x.isupper() else 0 for x in row]) for row in train_feat]
        cap_sd = np.std(cap_train)
        cap_train = [x / cap_sd for x in cap_train]
        train_feat_out = scipy.sparse.hstack((train_feat_out,np.array(cap_train)[:,None]))
        
        feat_name = feat_name + ["Char Count", "Word Count", "Uppercase Count"]
            
            
    clf.fit(train_feat_out, train_label)
    print("Positive Class :")
    informative = np.argsort(clf.coef_[0])[-10:]
    for i in informative:
        print(feat_name[i])
    
    print("\nNegative Class :")
    informative = np.argsort(clf.coef_[0])[0:10]
    for i in informative:
        print(feat_name[i])


# In[65]:


most_informative(MultinomialNB(), X_train, Y_train, char_feat=False)


# In[45]:


# add more data!
# read in dataset

def extend():
    global X_train, Y_train
    with open("Sentiment Analysis Dataset.csv", "r") as f:
        lines = f.readlines()

    newX = []
    newY = []

    for line in lines[1:]:
        newY.append(int(line.split(',')[1]))
        raw = ' '.join(line.rsplit()[1:])
        newX.append(raw)

    X_train = X_train + newX
    Y_train = Y_train + newY

