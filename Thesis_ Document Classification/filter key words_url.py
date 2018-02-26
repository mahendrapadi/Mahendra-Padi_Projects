import csv
import re
import os
import shutil


#+file = open("CC_Urls_2015_11_05.csv",'r')

#read= csv.reader(file)
#print len(read)

keywords= ['darts','pinbal','poker','blackjack'
]


lenofdataset=[]

id=0
y=0
directory = os.getcwd()
category='C:\Users\D065921\PycharmProjects\Thesis\Categories\Mixed'

#keyword='arts'
for keyword in keywords:
    x = 0
    #new_directory = os.path.join(category, keyword)
    #if os.path.exists(new_directory):
        #shutil.rmtree(new_directory)
    # os.mkdir(new_directory)
    # os.chdir(new_directory)
    #os.mkdir(category)
    os.chdir(category)

    with open(keyword + '.csv', 'w') as f:
        writer = csv.writer(f)

        os.chdir(directory)
        for root,dirs, files in os.walk(directory):

            os.chdir(directory)
            for file in files:

                print 'processing:', file
                #os.chdir(directory)

                if file.endswith(".csv"):

                    file_open = open(file, 'r')

                    read = csv.reader(file_open)

                    for row in read:
                            #lenofdataset.append(row)
                            string1 = "\\b"
                            string2 = "\\b"


                            word1 = ''.join((string1, keyword))
                            word2 = ''.join((word1,string2))

                            match = re.findall(str(word1), str(row))

                            #match = re.findall('\\bhome\\b', str(row))

                            if match:
                                rows = []
                                rows.append(row)
                                #rows.append([id])
                                # print rows
                                x=x+1
                                writer.writerows(rows)
                                #id = id + 1
                                #rows.append(row)


            print x
