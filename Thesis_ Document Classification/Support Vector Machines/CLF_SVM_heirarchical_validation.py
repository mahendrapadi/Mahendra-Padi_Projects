from sklearn.datasets import load_files
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn.grid_search import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier

from sklearn.linear_model import LogisticRegression
from operator import itemgetter

from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from sklearn.metrics.pairwise import cosine_similarity


from sklearn import svm
import numpy as np
import pandas
import pickle
import os
import sys
import nltk
import re
import shutil
from langdetect import detect



top_dir=os.listdir(os.getcwd())
start_dir=os.getcwd()


for dirx in top_dir:

    if dirx.endswith('.py'):
        x=0

    else:

        Path_main = os.path.join(start_dir,dirx)

        os.chdir(Path_main)

        filename  = open(dirx+"_outputfile_Heirarchy.txt",'w')
        sys.stdout = filename

        #          >>>>>>>>>>>>  Variable diclaration

        performance_init = 0
        performance_train = 0.0001

        performance_MNB = 0
        performance_LR = 0
        performance_SGD = 0


        Training_docs_yes = []
        Training_docs_no = []
        Prediction_docs = []

        performances_loop=[]


        X_train_MNB = 0
        X_train_LR = 0
        X_train_SGD = 0

        len_y = 11
        len_n =11


        iteration = 1
        loop = 0

        lang_check = 0

        # >>>>>>>>>>>>>>>>>>>>>  Getting Data Paths

        #Path_main=os.getcwd()

        path_test_gold = 0
        path_train = 0
        path_test = 0
        path_predict = 0
        categories = []

        sub_dirs=os.listdir(Path_main)

        #for root, dirs, files in os.walk(Path_main):



        for dir in sub_dirs:

              #if dir!='New folder' :
                if dir=='Gold':
                    path_test_gold =str(os.path.join(Path_main,'Gold'))
                    print "Gold_Path : ",path_test_gold
                #if dir=='Silver':
                #    path_test=str(os.path.join(Path_main,'Silver'))
                 #   print 'Silver_Path : ',path_test
                if dir=='Training':
                    path_train=str(os.path.join(Path_main,'Training'))
                    print 'Training_Path : ',path_train
                #if dir=='Unknown':
                 #   path_predict=os.path.join(Path_main,'Unknown')
                    #path_predict= str(os.path.normpath(Unknown_x + os.sep + os.pardir))
                  #  print 'Unknown_Path : ',path_predict

        for root,dirs,files in os.walk(path_train):
            for dir in dirs:
                categories.append(dir)
                #print dir
        print 'Categories : ',categories
        print 'yes_data : ',categories[0]
        print 'no_data : ',categories[1]


        # >>>>>>>>>>>>>>>>     Data preprocessing with NLTK

        stemmer = SnowballStemmer("english")
        wordnet_lemmatizer = WordNetLemmatizer()

        def tokenize_and_stem(text):
            # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
            tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
            filtered_tokens = []
            # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
            for token in tokens:
                if re.search(u'[a-zA-Z]', token):

                    filtered_tokens.append(token)
            #stems = [stemmer.stem(t) for t in filtered_tokens]

            lemmats =[wordnet_lemmatizer.lemmatize(t,pos='v') for t in filtered_tokens]
            stems = [stemmer.stem(k) for k in lemmats]
            #synonyms=[wordnet.synset(k) for k in lemmats]
            #print type(lemmats)
            #return stems
            return stems

        # >>>>>>>>>>>>>>   Loading the Data

        data_train = load_files(path_train, categories=categories)

        print '****************** Input *******************'

        print ('\n')

        Training_docs_yes.append(len(data_train.filenames))

        print("Training documents: %d " % len(data_train.filenames))

        print ('\n')

        # ##############        >>>>>>>>>>>>>>>     Vectorization
        vectorizer = TfidfVectorizer(sublinear_tf=True, lowercase=True, stop_words='english', min_df=3,max_df=.70,
                                         decode_error='ignore', strip_accents='ascii', tokenizer=tokenize_and_stem)

        # Train Data
        X_train = vectorizer.fit_transform(data_train.data)
        y_train = data_train.target


        # ################       >>>>>>>>>>>>     Classification

        print '********************** SGD Classifier *******************'

        #   >>>>>>>>>>  Grid Search

        alpha = [0.0001,0.001,0.01,0.1,1.0,2.0]
        average = [True,3,5,10]
        class_weight = [None,'balanced']
        epsilon = [0.1,0.01,0.5,0.0001,1]
            # eta0 = 0.0,
            # fit_intercept = True,
            # l1_ratio = 0.15,
            # learning_rate = 'optimal',
        loss = ['hinge']
        n_iter =[5,1]
        # n_jobs = 1,
        penalty = ['l2']
            # power_t = 0.5,
            # random_state = None,
            # shuffle = True,
            # verbose = 0, warm_start = False).fit(X_train, y_train)

        param_grid= dict(alpha=alpha, average=average, class_weight=class_weight, epsilon=epsilon,loss=loss, n_iter=n_iter,penalty=penalty)
        grid_SGD = GridSearchCV(SGDClassifier(), param_grid, cv=10, scoring='f1')

        # >>>>>>>>>>>  Model Training

        model_SGD= grid_SGD.fit(X_train,y_train)

        # >>>>>>>>>> Best Parameters on development set
        print '**********   Best parameters on Development set ********** '

        print model_SGD.best_params_
        print '\n'
        print '**********  Best Estimater which gave hei score  *********'

        print model_SGD.best_estimator_

        # >>>>>>   Saving the Model

        os.chdir(Path_main)

        Model_SGD = categories[0]+'_Model_Heirarchy_SGD.sav'
        pickle.dump(model_SGD, open(Model_SGD, 'wb'))

        Vector_SGD=categories[0]+'_Vector_Heirarchy_SGD.sav'
        os.chdir(Path_main)
        pickle.dump(vectorizer,open(Vector_SGD,'wb'))

        #print 'Saved training vector shape ', X_train_SGD.shape
        #y_train_SGD = data_train.target

        #  ------------------------------->>>>>>>>>>>>>>>>>>>>>>>>>>>>      Final Evaluation on Gold Standard

        print ('\n')
        print "*************************  Final Evaluation on Gold Standard   ************************"


        data_test_gold = load_files(path_test_gold, categories=categories)

        os.chdir(Path_main)

        Load_vector_SGD= pickle.load(open((categories[0]+'_Vector_Heirarchy_SGD.sav'),'rb'))

        X_test_gold_SGD = Load_vector_SGD.transform(data_test_gold.data)
        y_test_gold = data_test_gold.target
        
        # >>>>>>>>>>>> loading model from disk
        os.chdir(Path_main)
        Load_model_SGD = pickle.load(open((categories[0]+'_Model_Heirarchy_SGD.sav'),'rb'))

        # >>>>>>>>>>>>>>   Prediction on Gold standard data


        predict_test_gold_SGD = Load_model_SGD.predict(X_test_gold_SGD)

        accuracy_gold_SGD= metrics.accuracy_score(predict_test_gold_SGD, y_test_gold)

        print ('\n')

        print "Accuracy_SGD: ", accuracy_gold_SGD
        print '.'*50

        performance_gold_SGD = metrics.f1_score(predict_test_gold_SGD, y_test_gold)


        print "F1-score_SGD: ", performance_gold_SGD
        print '.'*50


        #score_roc_auc_scor_SGD = metrics.roc_auc_score(predict_test_gold_SGD, y_test_gold)

        #print 'roc_auc_score_SGD :', score_roc_auc_scor_SGD
        #print '.'*50


        print 'confusion matrics_SGD :'
        print metrics.confusion_matrix(predict_test_gold_SGD, y_test_gold)
