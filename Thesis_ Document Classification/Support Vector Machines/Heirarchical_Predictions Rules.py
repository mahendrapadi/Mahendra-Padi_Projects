from sklearn.datasets import load_files
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn import metrics

from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer

import pickle
import os
import sys
import nltk
import re
import shutil

from itertools import groupby
from operator import itemgetter


Path_unknown='C:\Users\D065921\Documents\Mahi\Thesis\Crawling\Crawled Data\last versions\Data_POOL_ver2\Final Prediction_HR_SGD\Unknown'
path_predicted_categories='C:\Users\D065921\Documents\Mahi\Thesis\Crawling\Crawled Data\last versions\Data_POOL_ver2\Final Prediction_HR_SGD\Predicted_categories_ver3'

#Path_unknown= raw_input('Enter the path of Unknown files which have to be categories')
#path_predicted_categories = raw_input('Enter the destination for predicted categories')

Data_Unknown_FP =load_files(Path_unknown)

#Result_Classifier1=[]


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
    # stems = [stemmer.stem(t) for t in filtered_tokens]

    lemmats = [wordnet_lemmatizer.lemmatize(t, pos='v') for t in filtered_tokens]
    stems = [stemmer.stem(k) for k in lemmats]
    # synonyms=[wordnet.synset(k) for k in lemmats]
    # print type(lemmats)
    # return stems
    return stems


Path_main =os.getcwd()
sub_dirs_level0=os.listdir(Path_main)

main_cat=['Arts','Business','Computers','Home','Recreation','Health','Science','Society','Sports','Games']

for dirs in sub_dirs_level0:

    print dirs

    if dirs.endswith('.py'):
        x = 0
    else:
        data_pool_Yes = []
        data_pool_No = []
        data_pool = []
        NOT_data_pool=[]
        category_pool = []

        path_sub_dirs_level1 = os.path.join(Path_main,dirs)
        sub_dirs_level1 = os.listdir(path_sub_dirs_level1)

        for dirs_level1 in sub_dirs_level1:

            #data_pool = []
            categories = []

            categories.append(dirs_level1)
            categories.append('Not_'+str(dirs_level1))

            category_pool.append(categories)

            #print categories

            path_classifier = os.path.join( path_sub_dirs_level1 , dirs_level1 )
            os.chdir(path_classifier)

            Load_vector = pickle.load(open((dirs_level1 + '_Vector_SGD.sav'), 'rb'))

            #print Load_vector

            X_Unknown_FP = Load_vector.transform(Data_Unknown_FP.data)
            #y_Unknown_FP = Data_Unknown.target

            # >>>>>>>>>>>> loading model from disk
            os.chdir(path_classifier)
            Load_model = pickle.load(open((dirs_level1 + '_Model_SGD.sav'), 'rb'))

            #print Load_model


            #categories = Load_vector.taget_names
            #print categories

            # >>>>>>>>>>>>>>   Prediction on Unknown_FP data

            predicted_Unknown = Load_model.predict(X_Unknown_FP)
            predicted_cnf_Unknown = Load_model.decision_function(X_Unknown_FP)

            # >>>>>>>>>>>>>> Zipping the Results

            for category, doc, conf in zip(predicted_Unknown, Data_Unknown_FP.filenames, predicted_cnf_Unknown):

                list_result = []  # temparay list to collect prediction results from all classifiers
                list_result.append(doc)

                list_result.append(categories[category])

                list_result.append(category)

                list_result.append(abs(conf))

                data_pool.append(list_result)

        data_pool.sort(key=lambda x: x[0], reverse=True)
        group_by = groupby(data_pool, itemgetter(0))

        prediction_list = [{'file': k, 'prediction': [[x[1],x[2], x[3]] for x in v]} for k, v in group_by]
        finalprediction_on_file=[]

        parent_cat = []
        child_cat = []
        #print 'categorypool',category_pool
        for i in category_pool:
            for j in main_cat:
                if j in i:
                    parent_cat = i  #append(i)
                    category_pool.remove(i)
        child_cat = category_pool
        #print'parent_cate', parent_cat
        #print'child', child_cat
        #print type(parent_cat)


        for file in prediction_list:
            chunk = file['prediction']
            #print 'chunk', chunk
            #print '*'*50
            k = sum([item[1] for item in chunk ])

            if k==4:
                for n in chunk:
                    x = []
                    x.append(file['file'])
                    x.append(n[0])
                    finalprediction_on_file.append(x)

                #print finalprediction_on_file

                #NOT_data_pool.append(file)
                #prediction_list.remove(file)
                #print NOT_data_pool
              #print file
            else:

                parent_says =0
                child_says=[]

                for item in chunk:

                    #print chunk

                    #print item[0]
                    #print parent_cat

                    if str(item[0]) in parent_cat:
                        #print '++++++++++++++++++'
                        parent_says=item ### item
                        #chunk.remove(item)

                    else:
                        child_says.append(item)

                #print 'parend_says', parent_says

                #print'child_says', child_says


                if (parent_says[1]== 0) and ( sum([item[1] for item in child_says]) == 3 ):

                    x=[]
                    x.append(file['file'])
                    x.append(parent_says[0])
                    finalprediction_on_file.append(x)

                    #prediction_list.remove(file)

                elif (parent_says[1] == 0) and ( sum([item[1] for item in child_says]) < 3 ):

                    child_says_yes=[]
                    for child in child_says:
                        if child[1]== 0:
                            child_says_yes.append(child)

                    if len(child_says_yes) == 1:
                        x = []
                        x.append(file['file'])
                        x.append(child_says_yes[0][0])
                        finalprediction_on_file.append(x)
                        x = []
                        x.append(file['file'])
                        x.append(parent_says[0])
                        finalprediction_on_file.append(x)
                    else:
                        if child_says_yes[0][2] > child_says_yes[1][2]:

                            x = []
                            x.append(file['file'])
                            x.append(child_says_yes[0][0])
                            finalprediction_on_file.append(x)
                            x = []
                            x.append(file['file'])
                            x.append(parent_says[0])
                            finalprediction_on_file.append(x)
                        else:
                            x = []
                            x.append(file['file'])
                            x.append(child_says_yes[1][0])
                            finalprediction_on_file.append(x)
                            x = []
                            x.append(file['file'])
                            x.append(parent_says[0])
                            finalprediction_on_file.append(x)

                elif (parent_says[1] == 1) and (sum([item[1] for item in child_says]) < 3):

                    child_says_yes=[]
                    child_says_no=[]

                    for l in child_says:
                        if l[1]==0:

                            child_says_yes.append(l)
                        else:
                            child_says_no.append(l)


                    for l in child_says_no:
                        x = []
                        x.append(file['file'])
                        x.append(l[0])
                        finalprediction_on_file.append(x)

                    y=[]
                    for e in child_says_yes:
                       y.append(e[2])

                    if parent_says[2] < max(y):

                        x = []
                        x.append(file['file'])
                        x.append(parent_cat[0])
                        finalprediction_on_file.append(x)

                        for k in child_says_yes:

                            if k[2]== max(y):

                                x = []
                                x.append(file['file'])
                                x.append(k[0])
                                finalprediction_on_file.append(x)

                    else:
                        x = []
                        x.append(file['file'])
                        #v=
                        x.append(parent_cat[1])
                        finalprediction_on_file.append(x)
                        for k in child_says:
                            for l in child_cat:
                                if k[0]==l[0]:
                                    x = []
                                    x.append(file['file'])
                                    x.append(l[1])
                                    finalprediction_on_file.append(x)

                                elif k[0]==l[1]:
                                    x = []
                                    x.append(file['file'])
                                    x.append(l[1])
                                    finalprediction_on_file.append(x)



        for i in finalprediction_on_file:
            #if i[0] in
            path_pred_cat = os.path.join(path_predicted_categories, dirs)

            if os.path.exists(path_pred_cat):

                file_final_dest = os.path.join(path_pred_cat, i[1])

                if os.path.exists(file_final_dest):
                    shutil.copy(i[0], file_final_dest)
                else:
                    os.mkdir(file_final_dest)
                    shutil.copy(i[0], file_final_dest)
            else:
                os.mkdir(path_pred_cat)
                file_final_dest = os.path.join(path_pred_cat, i[1])

                if os.path.exists(file_final_dest):
                    shutil.copy(i[0], file_final_dest)
                else:
                    os.mkdir(file_final_dest)
                    shutil.copy(i[0], file_final_dest)


'''
        for i in NOT_data_pool:
                        # if i[0] in
            path_pred_cat = os.path.join(path_predicted_categories, dirs)

            if os.path.exists(path_pred_cat):

                file_final_dest = os.path.join(path_pred_cat, i[1])

                if os.path.exists(file_final_dest):
                    shutil.copy(i[0], file_final_dest)
                else:
                    os.mkdir(file_final_dest)
                    shutil.copy(i[0], file_final_dest)
            else:
                os.mkdir(path_pred_cat)
                file_final_dest = os.path.join(path_pred_cat, i[1])

                if os.path.exists(file_final_dest):
                    shutil.copy(i[0], file_final_dest)
                else:
                    os.mkdir(file_final_dest)
                    shutil.copy(i[0], file_final_dest)
'''



                                #print i

        #print '*'*50
        #for i in NOT_data_pool:
          #  print i
        #print '*' * 50
        #for i in finalprediction_on_file_NOT:
         #   print i


'''
                elif (parent_says[1] == 1) and (sum([item[1] for item in child_says]) < 3):

                    y = []
                    for l in child_says:
                        if l[1] == 0:
                            y.append(l[2])

                    if parent_says[2] < max(y):

                        x = []
                        x.append(file['file'])
                        x.append(parent_cat[0])
                        finalprediction_on_file.append(x)

                    for k in child_says:
                        if k[1] == 0:
                            x = []
                            x.append(file['file'])
                            x.append(k[0])
                            finalprediction_on_file.append(x)


                    print 'parent_says',parent_says

                    print 'child_says',child_says



                    x=[]
                    x.append(file['file'])
                    x.append(parent_cat[0])
                    finalprediction_on_file.append(x)
                    for k in child_says:
                        x=[]
                        x.append(file['file'])
                        x.append(k[0])
                        finalprediction_on_file.append(x)



                elif (parent_says[1] == 1) and (sum([item[1] for item in child_says]) < 3):
                    x = []
                    x.append(file['file'])
                    x.append(parent_cat[0])
                    finalprediction_on_file.append(x)
                    for k in child_says:
                        if k[1]==0:
                            x=[]
                            x.append(file['file'])
                            x.append(k[0])
                            finalprediction_on_file.append(x)





        #finalprediction_on_file)
        print '*'*50
        for i in finalprediction_on_file:
            if i[0] in
            path_pred_cat = os.path.join(path_predicted_categories,dirs)
            if os.path.exists(path_pred_cat):
                file_final_dest=os.path.join(path_pred_cat,i[1])

                if os.path.exists(file_final_dest):
                    shutil.copy(i[0], file_final_dest)
                else:
                    os.mkdir(file_final_dest)
                    shutil.copy(i[0], file_final_dest)
            else:
                os.mkdir(path_pred_cat)
                file_final_dest = os.path.join(path_pred_cat, i[1])

                if os.path.exists(file_final_dest):
                    shutil.copy(i[0], file_final_dest)
                else:
                    os.mkdir(file_final_dest)
                    shutil.copy(i[0], file_final_dest)



            #print i



                        #parent_cat=item


            #else:



            #for i in  chunk:
             #   print [item for item in i]
              #  #for item in i]
                #print k


        #for i in list:
            #print  i

'''






'''

                if category == 0:
                    list_result.append(doc)

                    list_result.append(categories[category])

                    list_result.append(category)

                    list_result.append(conf)

                    data_pool_Yes.append(list_result)
                else:
                    list_result.append(doc)
                    list_result.append(categories[category])
                    list_result.append(category)
                    list_result.append(conf)
                    data_pool_No.append(list_result)

        data_pool_Yes.sort(key=lambda x: x[0], reverse=True)
        data_pool_No.sort(key=lambda x:x[0],reverse=True)

        group_Yes = groupby(data_pool_Yes, itemgetter(0))
        group_No = groupby(data_pool_No,itemgetter(0))

        list_Yes= [{'file': k, 'prediction': [[x[1], x[3]] for x in v] } for k, v in group_Yes]
        list_No = [{'file': k, 'prediction': [[x[1], x[3]] for x in v]} for k, v in group_No]


        #print  [[item for item in data] for (key, data) in groups]

        for item in list_Yes:

            k = item['prediction']


            if len(k) == 1:

                predicted_main_path = os.path.join(path_predicted_categories,dirs)
                if os.path.exists(predicted_main_path):
                    predicted_cat_path = os.path.join(predicted_main_path, k[0][0])
                    if os.path.exists((predicted_cat_path)):
                        shutil.copy(item['file'],predicted_cat_path)
                    else:

                        os.mkdir(predicted_cat_path)
                        shutil.copy(item['file'], predicted_cat_path)
                else:
                    os.mkdir(predicted_main_path)
                    predicted_cat_path = os.path.join(predicted_main_path, k[0][0])
                    shutil.copy(item['file'], predicted_cat_path)

            elif len(k) == 2:

                if k[0][1]>k[1][1]:
                    predicted_main_path = os.path.join(path_predicted_categories, dirs)
                    if os.path.exists(predicted_main_path):
                        predicted_cat_path = os.path.join(predicted_main_path, k[0][0])
                        if os.path.exists((predicted_cat_path)):
                            shutil.copy(item['file'], predicted_cat_path)
                        else:

                            os.mkdir(predicted_cat_path)
                            shutil.copy(item['file'], predicted_cat_path)
                    else:
                        os.mkdir(predicted_main_path)
                        predicted_cat_path = os.path.join(predicted_main_path, k[0][0])
                        shutil.copy(item['file'], predicted_cat_path)
                else:

                    predicted_main_path = os.path.join(path_predicted_categories, dirs)
                    if os.path.exists(predicted_main_path):
                        predicted_cat_path = os.path.join(predicted_main_path, k[1][0])
                        if os.path.exists((predicted_cat_path)):
                            shutil.copy(item['file'], predicted_cat_path)
                        else:

                            os.mkdir(predicted_cat_path)
                            shutil.copy(item['file'], predicted_cat_path)
                    else:
                        os.mkdir(predicted_main_path)
                        predicted_cat_path = os.path.join(predicted_main_path, k[1][0])
                        shutil.copy(item['file'], predicted_cat_path)

            elif len(k) == 3 or len(k) == 4:

                #print item['file']
                #print k
                pred_cat= min(item1[0] for item1 in k)
                predicted_main_path = os.path.join(path_predicted_categories, dirs)
                if os.path.exists(predicted_main_path):
                    predicted_cat_path = os.path.join(predicted_main_path, pred_cat)
                    if os.path.exists((predicted_cat_path)):
                        shutil.copy(item['file'], predicted_cat_path)
                    else:

                        os.mkdir(predicted_cat_path)
                        shutil.copy(item['file'], predicted_cat_path)
                else:
                    os.mkdir(predicted_main_path)
                    predicted_cat_path = os.path.join(predicted_main_path, pred_cat)
                    shutil.copy(item['file'], predicted_cat_path)

            for item in list_No:

                k = item['prediction']

                if len(k) == 1:

                    predicted_main_path = os.path.join(path_predicted_categories, dirs)
                    if os.path.exists(predicted_main_path):
                        predicted_cat_path = os.path.join(predicted_main_path, k[0][0])
                        if os.path.exists((predicted_cat_path)):
                            shutil.copy(item['file'], predicted_cat_path)
                        else:

                            os.mkdir(predicted_cat_path)
                            shutil.copy(item['file'], predicted_cat_path)
                    else:
                        os.mkdir(predicted_main_path)
                        predicted_cat_path = os.path.join(predicted_main_path, k[0][0])
                        shutil.copy(item['file'], predicted_cat_path)

                elif len(k) == 2:

                    if k[0][1] > k[1][1]:
                        predicted_main_path = os.path.join(path_predicted_categories, dirs)
                        if os.path.exists(predicted_main_path):
                            predicted_cat_path = os.path.join(predicted_main_path, k[0][0])
                            if os.path.exists((predicted_cat_path)):
                                shutil.copy(item['file'], predicted_cat_path)

                            else:

                                os.mkdir(predicted_cat_path)
                                shutil.copy(item['file'], predicted_cat_path)
                        else:
                            os.mkdir(predicted_main_path)
                            predicted_cat_path = os.path.join(predicted_main_path, k[0][0])
                            shutil.copy(item['file'], predicted_cat_path)
                    else:

                        predicted_main_path = os.path.join(path_predicted_categories, dirs)
                        if os.path.exists(predicted_main_path):
                            predicted_cat_path = os.path.join(predicted_main_path, k[1][0])
                            if os.path.exists((predicted_cat_path)):
                                shutil.copy(item['file'], predicted_cat_path)
                            else:

                                os.mkdir(predicted_cat_path)
                                shutil.copy(item['file'], predicted_cat_path)
                        else:
                            os.mkdir(predicted_main_path)
                            predicted_cat_path = os.path.join(predicted_main_path, k[1][0])
                            shutil.copy(item['file'], predicted_cat_path)

                elif len(k) == 3 or len(k) == 4:

                    # print item['file']
                    # print k
                    pred_cat = min(item1[0] for item1 in k)
                    predicted_main_path = os.path.join(path_predicted_categories, dirs)
                    if os.path.exists(predicted_main_path):
                        predicted_cat_path = os.path.join(predicted_main_path, pred_cat)
                        if os.path.exists((predicted_cat_path)):
                            shutil.copy(item['file'], predicted_cat_path)
                        else:

                            os.mkdir(predicted_cat_path)
                            shutil.copy(item['file'], predicted_cat_path)
                    else:
                        os.mkdir(predicted_main_path)
                        predicted_cat_path = os.path.join(predicted_main_path, pred_cat)
                        shutil.copy(item['file'], predicted_cat_path)





                                #print max_val


                #print'---------------------'

                #print k[1]

    print '*' * 100

                #os.chdir(predicted_main_path)

                #predicted_cat_path=os.join.path(predicted_main_path,k[0])


                #print item['file'], item['prediction']


        #print groups






        for i in data_pool_Yes:
            z = []
            z.append(i[0])

            for j in data_pool_Yes:
                y = []
                c=[]
                if i[0] == j[0]:

                    if j[1] not in c :
                        c.append(j[1])

                        y.append(j[1])
                        y. append(j[2])
                z.append(y)

            x.append(z)


        for l in x:
            print l




            if data_pool_Yes[i][0] == data_pool_Yes[(i+1)][0]:

                x.append(i)

        for l in x:
            print l




                print

                print i
        print '*' * 100
        for k in data_pool_No:
            print k




#data_pool.sort(key=lambda x: x[0], reverse=True)
#for i in data_pool:
 #   print i





for row in data_pool:
    i=0
    j=1

    temp_list = []
    temp_list.append(row[0])

    for i in range(len(data_pool)):

        if row[i][0] == row[i+1][0]:
            temp_list.append()

            print i
                #print dirs_level1

'''
