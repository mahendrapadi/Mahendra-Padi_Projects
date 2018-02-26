# thesis_documentclassification
Document Classification

OverView: 

In order to achieve the goal of the thesis, It was organized in three different phases.
Phase 1. Data collection.
Phase 2. Supervised Learning with self learning approach.
Phase 3. Hierarchical Classification

Data sets:

 Five Different data sets have been prepared and used for the reasearch.
 
 1. Training dataset: Which is used to train the classiﬁers. Initially, it has very limited ammount of annotated documents. 
 2. Development dataset:Is a validation dataset to estimate the performance of a classiﬁer during the self-learning in Phase 2.It is partially annotated. 
 
 3. Gold standard dataset:Is the Finalevaluation dataset. All the examples in this dataset are labeled manually. 
 
 4. Unknown dataset 1: Used in Phase 2. In self-learning approach, the classiﬁer makes predictions on this dataset and add the documents(which has the most confident values) to the training data set after every iteration. No Class labels.
 
 5. Unknown dataset 2:Used in the  hierarchical classiﬁcation (Phase 3 ). No Class labels.


Python Files: 

In Phase 1 :

    1. CC_Url_Index_extraction: Impleted to extract the URLs from the Common crawl URL Index.
 
    2. filter key words_url : Implemented to filter the URLs with a keyword(category name). 
 
    3. Text_crawler_ver1: Implemented to extract the text data from a URL. This script reads all URLs one by one from a CSV file inorder
       to crawl the text data from it.
       #s 
 In Phase 2 : (You can find the below files in the spacific algorithem's folder) 
 
    4. CLF_MNB : This scripts was implemented for Supervised classification with self learning approach (in phase 2) with Multinomial
       Naïve Bayes.
 
    5. CLF_SVM: This scripts was implemented for classification with self learning approach (in phase 2) with Support Vector Machines. 
    
  In Phase 2 :
  
    6. Hierarchical_Predictions_Rules: This Script was implemented for Hierarchical Classification(in 3rd Phase). This Script will uses
       the models which were developed in supervised learnig phase(2nd phase). All the models of a category class will apply on unknown
       dataset2 and bsed on their predictions of a perticular document, then this script applies hierarchical rules on predictions and
       make changes to the pridections according to the rules.
   
   Final Validation
 
    7. CLF_MNB_hierarchical_ validation:  This script validate the final results after Hierarchical classification on Gold Standard data
       set.  IOt uses the classifier Multinomial Naïve Bayes
 
    8. CLF_SVM_heirarchical_validation: This script validate the final results after Hierarchical classification on Gold Standard data
       set.  IOt uses the classifier Support Vector Machines.
    
