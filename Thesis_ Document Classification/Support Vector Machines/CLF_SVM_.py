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
            #stems = [stemmer.stem(t) for t in filtered_tokens]

            lemmats =[wordnet_lemmatizer.lemmatize(t,pos='v') for t in filtered_tokens]
            stems = [stemmer.stem(k) for k in lemmats]
            #synonyms=[wordnet.synset(k) for k in lemmats]
            #print type(lemmats)
            #return stems
            return stems


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

        # #################################################################################

        # >>>>>>>>>>>>>>>>>>>>>>  Semi Supervised Learning

        #while  performance_train > performance_init:

         #while loop < (len_y<10) or (len_n<10):

        #while (len_y > 10) or (len_n > 10):
        while loop<3:

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
            #vectorizer = TfidfVectorizer(sublinear_tf=True, min_df=3,max_df=.80, lowercase=True, stop_words='english',decode_error='replace',tokenizer=tokenize_and_stem)
            vectorizer = TfidfVectorizer(sublinear_tf=True, lowercase=True, stop_words='english', min_df=3,max_df=.70,
                                         decode_error='ignore', strip_accents='ascii', tokenizer=tokenize_and_stem)

            # Train Data
            X_train = vectorizer.fit_transform(data_train.data)
            y_train = data_train.target

            print'train data shape: ', X_train.shape

            #print 'Train Features: ', vectorizer.get_feature_names()

            # Test Data
            X_test = vectorizer.transform(data_test.data)
            y_test = data_test.target

            print 'test data shape : ',X_test.shape


            # Unknown data
            X_predict = vectorizer.transform(data_predict.data)

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

            #model_SGD = SGDClassifier(alpha=0.0001, average=False, class_weight=None, epsilon=0.1,eta0=0.0, fit_intercept=True, l1_ratio=0.15,learning_rate='optimal', loss='hinge', n_iter=5, n_jobs=1, penalty='l2', power_t=0.5, random_state=None, shuffle=True,verbose=0, warm_start=False).fit(X_train, y_train)

            model_SGD= grid_SGD.fit(X_train,y_train)

            #  >>>>>>>>>>>   Saving the Model

            # Model_SGD = 'Model_SGD.sav'
            # pickle.dump(model_SGD, open(Model_SGD, 'wb'))

            # Load_model_SGD = pickle.load(open(Model_SGD, 'rb'))

            # >>>>>>>>>>>    Prediction on test data

            predict_test3 = model_SGD.predict(X_test)
            accuracy_SGD = metrics.accuracy_score(predict_test3, y_test)
            print ('\n')
            print "Accuracy_SGD: ", accuracy_SGD

            performance_train_SGD = metrics.f1_score(predict_test3, y_test)
            performances_loop.append(performance_train_SGD)
            print "F1-score_SGD: ", performance_train_SGD

            #score_roc_auc_scor_SGD = metrics.roc_auc_score(predict_test3, y_test)
            #print 'roc_auc_score_SGD:', score_roc_auc_scor_SGD

            print 'confusion matrics_SGD:'
            print metrics.confusion_matrix(predict_test3, y_test)
            print ('\n')
            print '-' * 100

            # >>>>> predictions on unknown data

            predicted_SGD = model_SGD.predict(X_predict)
            predicted_cnf_SGD = model_SGD.decision_function(X_predict)

            # >>>>>>   Saving the Model

            if performance_train_SGD > performance_SGD:
                os.chdir(Path_main)

                Model_SGD = categories[0]+'_Model_SGD.sav'
                pickle.dump(model_SGD, open(Model_SGD, 'wb'))

                performance_SGD = performance_train_SGD

                # >>>>  Saving the Vectorizer according to the Saved model. We use this vectoriser to transform Gold evaluation dataset

                #vectorizer_SGD = TfidfVectorizer(sublinear_tf=True, min_df=3, lowercase=True, stop_words='english',decode_error='replace',analyzer='word')  # tokenizer=tokenize_and_stem)
                #vectorizer_SGD = TfidfVectorizer(sublinear_tf=True, min_df=3,max_df=.80, lowercase=True, stop_words='english',decode_error='replace', tokenizer=tokenize_and_stem)

                vectorizer_SGD= TfidfVectorizer(sublinear_tf=True, lowercase=True, stop_words='english', min_df=3,max_df=.70,
                                             decode_error='ignore', strip_accents='ascii', tokenizer=tokenize_and_stem)

                # vectorisation of the training data set of the above saved modal
                X_train_SGD = vectorizer_SGD.fit_transform(data_train.data)

                Vector_SGD=categories[0]+'_Vector_SGD.sav'
                os.chdir(Path_main)
                pickle.dump(vectorizer_SGD,open(Vector_SGD,'wb'))

                print 'Saved training vector shape ', X_train_SGD.shape
                y_train_SGD = data_train.target


            #performance.append(performance_list)  # appending the performance_list which has the performances of the above classifiers

            performance_train = max(performances_loop)   # taking the maximum performance from the performance_list

            prediction_datapool = []

            yes_datapool = []
            no_datapool = []

            data_pool = []

            # ###################################################          >>>>>>>>>>>>>>>>     Zipping the Results

            for category_SGD, doc,conf_SGD in zip(predicted_SGD,data_predict.filenames,predicted_cnf_SGD):

                list_result = []  # temparay list to collect prediction results from all classifiers

                list_result.append(doc)

                list_result.append(data_train.target_names[category_SGD])

                list_result.append(conf_SGD)

                data_pool.append(list_result)

            for i in data_pool:
                if i[1]== categories[0]:
                    yes_datapool.append(i)
                else:
                    no_datapool.append(i)


            #     >>>>>>>>>  Sorting both Datapools based on the highest confidence score of the classifiers


            yes_datapool.sort(key=lambda x: x[2], reverse=True)
            no_datapool.sort(key=lambda x: x[2], reverse=False)


            #print yes_datapool


            print 'length of Yes_dataPool',len(yes_datapool)
            print 'length of No_dataPool',len(no_datapool)

            #high_conf_Y=yes_datapool[:1]
            #high_conf_N=no_datapool[:1]


            #print 'Highest confident from YES data pool',high_conf_Y[2]
            #print 'Highest confident from NO data pool',high_conf_N[2]



                #print no_datapool

            # ######################    >>>>>>>>>>>>>>>>>>>>     getting the directory paths of training data

            directory_train_yes = os.path.join(path_train,categories[0])
            directory_train_no = os.path.join(path_train, categories[1])

            # list1 = []
            ele = 0
            len_y = len(yes_datapool)
            pick_y = len_y*0.05
            cutoff_y= int((yes_datapool[0][2]*-1)+1)
            print yes_datapool[0][2]
            #print int((yes_datapool[0][2]*-1))
            #print cutoff_y
            #print yes_datapool[:int(len_y)]

            for i in range(0, int(pick_y)):
            #for i in range(0,20):
                #if ele < len_y:
                #if ((yes_datapool[i][2][0]) > .75):
                    #for j in range(0, 10):
                    #if ((yes_datapool[i][2])*-1)< cutoff_y:
                        shutil.copy(yes_datapool[i][0], directory_train_yes)
                        os.remove(yes_datapool[i][0])
                        #yes_datapool.remove(yes_datapool[i])
                        #len_y=len_y-1
                        ele = ele+1
                #else:
                       # break

            print "Yes_documents moved from un know to training : ", ele

            ele = 0
            len_n = len(no_datapool)
            pick_n= len_n*0.05
            #print no_datapool[:int(len_n)]
            #cutoff_n= int(no_datapool[0][2])+1
            #print cutoff_n

            for i in range(0, int(pick_n)):
            #for i in range(0,20):
                #if ele < len_n:
                    #if ((no_datapool[i][2][1]) > .60):
                            # for j in range(0, 10):
                        #if (no_datapool[i][2]) < cutoff_n:
                            shutil.copy(no_datapool[i][0], directory_train_no)
                            os.remove(no_datapool[i][0])
                            #no_datapool.remove( no_datapool[i])
                            # len_n=len_n-1
                            ele = ele + 1

                #else:
                        #break
            print "No_documents moved from unknown to training : " , ele

        print "#"*100
        print '\n'
        print 'Training finished....'
        print 'Train docs: '

        print Training_docs_yes

        print 'Prediction docs : '
        print Prediction_docs

        print 'Final Performance : '

        print performances_loop
        print '\n'

        print '*'*70
        print 'Best performanvces of the classifiers on Silver standard Data :'
        #print 'Permance_MNB : ',performance_MNB
        #print 'Performance_LR : ',performance_LR
        print 'Performance_SGD : ',performance_SGD
        print '*'*70



        #  ------------------------------->>>>>>>>>>>>>>>>>>>>>>>>>>>>      Final Evaluation on Gold Standard

        print ('\n')
        print "*************************  Final Evaluation on Gold Standard   ************************"


        data_test_gold = load_files(path_test_gold, categories=categories, )
        os.chdir(Path_main)

        Load_vector_SGD= pickle.load(open((categories[0]+'_Vector_SGD.sav'),'rb'))

        X_test_gold_SGD = Load_vector_SGD.transform(data_test_gold.data)
        y_test_gold = data_test_gold.target
        
        # >>>>>>>>>>>> loading model from disk
        os.chdir(Path_main)
        Load_model_SGD = pickle.load(open((categories[0]+'_Model_SGD.sav'),'rb'))

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
