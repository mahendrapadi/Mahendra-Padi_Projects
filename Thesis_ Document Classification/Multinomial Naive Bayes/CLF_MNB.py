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

        filename  = open(dirx+"_outputfile.txt",'w')
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

        #performances_loop=[]
        performances_loop = []


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
                if dir=='Silver':
                    path_test=str(os.path.join(Path_main,'Silver'))
                    print 'Silver_Path : ',path_test
                if dir=='Training':
                    path_train=str(os.path.join(Path_main,'Training'))
                    print 'Training_Path : ',path_train
                if dir=='Unknown':
                    path_predict=os.path.join(Path_main,'Unknown')
                    #path_predict= str(os.path.normpath(Unknown_x + os.sep + os.pardir))
                    print 'Unknown_Path : ',path_predict

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
            stems = [stemmer.stem(t) for t in filtered_tokens]

            #lemmats =[wordnet_lemmatizer.lemmatize(t,pos='v') for t in filtered_tokens]
            #stems = [stemmer.stem(k) for k in lemmats]
            #synonyms=[wordnet.synset(k) for k in lemmats]
            #print type(lemmats)
            #return stems
            return stems
	'''
        # >>>>>>>>>>>>>>>>>  Calculating Cosine similarity btn Unknown and Silver data and getting YES_documents for training



        data_silver_yes = load_files(path_test, categories=categories[0])



        data_unknown=load_files(path_predict)


        #print data_unknown.filenames


        #vectorizer = TfidfVectorizer(sublinear_tf=True, min_df=3,max_df=.80, lowercase=True, stop_words='english',decode_error='replace',tokenizer=tokenize_and_stem)
        vectorizer = TfidfVectorizer(sublinear_tf=True, lowercase=True, stop_words='english',min_df=3,max_df=.80,
                                         decode_error='ignore',strip_accents='ascii', tokenizer=tokenize_and_stem)



        # >>>>> Yes Data
        silver_yes_vect = vectorizer.fit_transform(data_silver_yes.data).toarray()
        unknown_vect = vectorizer.transform((data_unknown.data)).toarray()

        tfidf_silver_yes = np.sum(silver_yes_vect, axis=0)

        cosine = cosine_similarity(unknown_vect[0:len(unknown_vect)], tfidf_silver_yes)

        #print cosine
        unknown_sim_score=[]
        for doc,sim_score in zip(data_unknown.filenames,cosine):

            x=[]
            x.append(doc)
            x.append(sim_score)

            unknown_sim_score.append(x)

        unknown_sim_score.sort(key=lambda x: x[1], reverse=True)

        path_train_yes=os.path.join(path_train,categories[0])

        for i in range (0,100):
            shutil.copy(unknown_sim_score[i][0], path_train_yes)
            os.remove(unknown_sim_score[i][0])


        # >>>>>> No Data
        data_silver_no = load_files(path_test, categories=categories[1])
        data_unknown = load_files(path_predict)


        silver_no_vect = vectorizer.fit_transform(data_silver_no.data).toarray()
        unknown_vect = vectorizer.transform((data_unknown.data)).toarray()

        tfidf_silver_no = np.sum(silver_no_vect, axis=0)

        cosine = cosine_similarity(unknown_vect[0:len(unknown_vect)], tfidf_silver_no)

        #print cosine
        unknown_sim_score=[]
        for doc,sim_score in zip(data_unknown.filenames,cosine):

            x=[]
            x.append(doc)
            x.append(sim_score)

            unknown_sim_score.append(x)

        unknown_sim_score.sort(key=lambda x: x[1], reverse=True)

        path_train_no = os.path.join(path_train,categories[1])

        for i in range (0,100):
            shutil.copy(unknown_sim_score[i][0], path_train_no)
            os.remove(unknown_sim_score[i][0])
    '''


        # #################################################################################

        # >>>>>>>>>>>>>>>>>>>>>>  Semi Supervised Learning

        #while  performance_train > performance_init:



        #performance_loop= []

        while loop < 5:
            loop=loop+1

            performance_init = performance_train


            # >>>>>>>>>>>>>>   Loading the Data

            data_train = load_files(path_train, categories=categories)
            data_test = load_files(path_test, categories=categories)
            data_predict = load_files(path_predict)


            print '****************** Input *******************'

            print ('\n')

            print ' In Iteration : ',iteration
            iteration = iteration+1
            Training_docs_yes.append(len(data_train.filenames))

            print("Training documents: %d " % len(data_train.filenames))
            print("Testing(Silver standard) documents : %d " % len(data_test.filenames))
            Prediction_docs.append(len(data_predict.filenames))
            print("Documents for prediction %d " % len(data_predict.filenames))
            print("Number of categories : %d " % len(data_train.target_names)), data_train.target_names

            print ('\n')

            # ##############        >>>>>>>>>>>>>>>     Vectorization


            #vectorizer = TfidfVectorizer(sublinear_tf=True, min_df=3, lowercase=True, stop_words='english',decode_error='replace',analyzer='word')  # tokenizer=tokenize_and_stem)
            #vectorizer = TfidfVectorizer(sublinear_tf=True, min_df=3,max_df=.80, lowercase=True, stop_words='english',decode_error='replace',tokenizer=tokenize_and_stem,max_features=10000)


            vectorizer = TfidfVectorizer(sublinear_tf=True, lowercase=True, stop_words='english', min_df=3, max_df=.80,
                                         decode_error='ignore', strip_accents='ascii', tokenizer=tokenize_and_stem)

            # Train Data
            X_train = vectorizer.fit_transform(data_train.data)
            y_train = data_train.target
            print'train data shape: ', X_train.shape

            # Test Data
            X_test = vectorizer.transform(data_test.data)
            y_test = data_test.target
            print 'test data shape : ', X_test.shape

            # Unknown data
            X_predict = vectorizer.transform(data_predict.data)

            # ################       >>>>>>>>>>>>     Classification

            print '********************** Multinominal NB *******************'

            # >>>>>>>>>>>    Grid Search

            alpha = [0.0001,0.001,0.01,0.1, 1,10,100,1000]
            #cross_valid=5,10,15

            param_grid = dict(alpha=alpha)
            #param_grid_cv=dict(cv=cross_valid)

            grid_MNB = GridSearchCV(MultinomialNB(), param_grid, cv=10, scoring='f1')

            #grid_MNB = GridSearchCV(MultinomialNB(), param_grid, cv=10, scoring='f1')

            # >>>>>>>>>>>>>     Model Training

            # model_MNB = LogisticRegression(penalty='l2', dual=True, tol=0.0001, C=1, fit_intercept=True, intercept_scaling=1.0,class_weight=None, random_state=None).fit(X_train, y_train)

            model_MNB = grid_MNB.fit(X_train, y_train)

            # >>>>>>>>>> Best Parameters on development set
            print '**********   Best parameters on Development set ********** '

            print model_MNB.best_params_
            print '\n'
            print '**********  Best Estimater which gave hei score  *********'

            print model_MNB.best_estimator_


            # >>>>>>>>>>>>>      Prediction on test data

            predict_test2 = model_MNB.predict(X_test)
            accuracy_MNB = metrics.accuracy_score(predict_test2, y_test)
            print ('\n')
            print "Accuracy_MNB: ", accuracy_MNB

            performance_train_MNB = metrics.f1_score(predict_test2, y_test)
            performances_loop.append(performance_train_MNB)
            print "F1-score_MNB: ", performance_train_MNB

            #score_roc_auc_scor_MNB = metrics.roc_auc_score(predict_test2, y_test)
            #print 'roc_auc_score_MNB:', score_roc_auc_scor_MNB

            print 'confusion matrics_MNB:'
            print metrics.confusion_matrix(predict_test2, y_test)
            print ('\n')

            # >>>>>>>>>>>>     Predictions on unknown data

            predicted_MNB = model_MNB.predict(X_predict)
            predicted_cnf_MNB = model_MNB.predict_proba(X_predict)

            # >>>>>>>>>>>>>   Saving the Model

            if performance_train_MNB > performance_MNB:
                Model_MNB = categories[0] + '_Model_MNB.sav'
                pickle.dump(model_MNB, open(Model_MNB, 'wb'))

                performance_MNB = performance_train_MNB

                # >>>>>>>>>>>> Saving the Vectorizer according to the Saved model. We use this vectoriser to transform Gold evaluation dataset

                #vectorizer_MNB = TfidfVectorizer(sublinear_tf=True, min_df=3, max_df=0.70, lowercase=True, stop_words='english',decode_error='replace', analyzer='word')  # tokenizer=tokenize_and_stem)
                #

                #vectorizer_MNB = TfidfVectorizer(sublinear_tf=True, min_df=3,max_df=.80, lowercase=True, stop_words='english',decode_error='replace',tokenizer=tokenize_and_stem,max_features=10000)

                vectorizer_MNB = TfidfVectorizer(sublinear_tf=True, lowercase=True, stop_words='english', min_df=3,
                                             max_df=.80,
                                             decode_error='ignore', strip_accents='ascii', tokenizer=tokenize_and_stem)

                # vectorisation of the training dada set of the above saved modal
                X_train_MNB = vectorizer_MNB.fit_transform(data_train.data)

                Vector_MNB = categories[0] + '_Vector_MNB.sav'
                pickle.dump(vectorizer_MNB, open(Vector_MNB, 'wb'))

                print 'Saved training vector shape ', X_train_MNB.shape

                y_train_MNB = data_train.target

            performance_train = max(performances_loop)  # taking the maximum performance from the performance_list

            prediction_datapool = []

            yes_datapool = []
            no_datapool = []

            data_pool = []

            # ###################################################          >>>>>>>>>>>>>>>>     Zipping the Results

            for category_MNB, doc, conf_MNB in zip(predicted_MNB, data_predict.filenames, predicted_cnf_MNB):
                list_result = []  # temparay list to collect prediction results from all classifiers

                list_result.append(doc)

                list_result.append(data_train.target_names[category_MNB])

                list_result.append(conf_MNB)
                data_pool.append(list_result)

            for i in data_pool:
                if i[1] == categories[0]:
                    yes_datapool.append(i)
                else:
                    no_datapool.append(i)

            # >>>>>>>>>  Sorting both Datapools based on the highest confidence score of the classifiers


            yes_datapool.sort(key=lambda x: x[2][0], reverse=True)
            no_datapool.sort(key=lambda x: x[2][1], reverse=True)

            print len(yes_datapool)
            print len(no_datapool)

            # ######################    >>>>>>>>>>>>>>>>>>>>     getting the directory paths of training data

            directory_train_yes = os.path.join(path_train, categories[0])
            directory_train_no = os.path.join(path_train, categories[1])

            # list1 = []
            ele = 0
            len_y = len(yes_datapool) * 0.05
            #print yes_datapool[:int(len_y)]

            for i in range(0, int(len_y)):
                # for i in range(0,20):
                # if ele < len_y:
                # if ((yes_datapool[i][2][0]) > .75):
                # for j in range(0, 10):
                shutil.copy(yes_datapool[i][0], directory_train_yes)
                os.remove(yes_datapool[i][0])
                yes_datapool.remove(yes_datapool[i])
                # len_y=len_y-1
                ele = ele + 1
                # else:
                # break

            print "Yes_documents moved from un know to training : ", ele

            ele = 0
            len_n = len(no_datapool) * 0.05
            #print no_datapool[:int(len_n)]

            for i in range(0, int(len_n)):
                # for i in range(0,20):
                # if ele < len_n:
                # if ((no_datapool[i][2][1]) > .60):
                # for j in range(0, 10):
                shutil.copy(no_datapool[i][0], directory_train_no)
                os.remove(no_datapool[i][0])
                no_datapool.remove(no_datapool[i])
                # len_n=len_n-1
                ele = ele + 1

                # else:
                # break
            print "No_documents moved from unknown to training : ", ele

        print "#" * 100
        print '\n'
        print 'Training finished....'
        print 'Train docs: '

        print Training_docs_yes

        print 'Prediction docs : '
        print Prediction_docs

        print 'Final Performance : '

        print performances_loop
        print '\n'

        print '*' * 70
        print 'Best performanvces of the classifiers on Silver standard Data :'
        print 'Performance_MNB : ', performance_MNB
        print '*' * 70
        #  ------------------------------->>>>>>>>>>>>>>>>>>>>>>>>>>>>      Final Evaluation on Gold Standard

        print ('\n')
        print "*************************  Final Evaluation on Gold Standard   ************************"


        data_test_gold = load_files(path_test_gold, categories=categories, )

        Load_vector_MNB= pickle.load(open((categories[0]+'_Vector_MNB.sav'),'rb'))

        X_test_gold_MNB = Load_vector_MNB.transform(data_test_gold.data)

        print 'gold data vector : ',X_test_gold_MNB.shape
        y_test_gold = data_test_gold.target

        # >>>>>>>>>>>> loading model from disk

        Load_model_MNB = pickle.load(open((categories[0]+'_Model_MNB.sav'),'rb'))

        # >>>>>>>>>>>>>>   Prediction on Gold standard data


        predict_test_gold_MNB = Load_model_MNB.predict(X_test_gold_MNB)


        accuracy_gold_MNB= metrics.accuracy_score(predict_test_gold_MNB, y_test_gold)

        print ('\n')

        print "Accuracy_MNB: ", accuracy_gold_MNB
        print '.'*50

        performance_gold_MNB = metrics.f1_score(predict_test_gold_MNB, y_test_gold)


        print "F1-score_MNB: ", performance_gold_MNB
        print '.'*50


        #score_roc_auc_scor_MNB = metrics.roc_auc_score(predict_test_gold_MNB, y_test_gold)

        #print 'roc_auc_score_MNB :', score_roc_auc_scor_MNB
        #print '.'*50


        print 'confusion matrics_MNB :'
        print metrics.confusion_matrix(predict_test_gold_MNB, y_test_gold)

