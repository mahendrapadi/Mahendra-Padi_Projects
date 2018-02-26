import csv
import requests
from bs4 import BeautifulSoup
from itertools import izip
import justext
import os
from os.path import basename
from sys import getsizeof

#txt=[]
#urls=[]
dir1= os.getcwd()


for root, dirs, files in os.walk(dir1):

    for file in files:

        if file.endswith('.csv'):

            #os.chdir(dir)

            with open(file) as f:
                csv_reader= csv.reader(f,delimiter=";")
                print dir1

                dir2 = dir1 + "\\" + basename(file)+"docs"
                print dir2

                os.mkdir(dir2)
                os.chdir(dir2)

                for row in csv_reader:
                       # urls=i
                       txt = []
                       #txt.append("url:")
                       #txt.append(str(row[0]))

                       #txt.append("\n")
                       try:

                           r = requests.get(row[0])
                           #paragraphs = justext.justext(r.content, justext.get_stoplist("English"))
                           paragraphs = justext.justext(r.content, justext.get_stoplist("English"),max_heading_distance=200, no_headings= False)

                           for paragraph in paragraphs:
                                if not paragraph.is_boilerplate:

                                    txt.append(paragraph.text)
                                    txt.append("\n")
                                   
                           if getsizeof(txt)>100:
                                with open(str(row[1]) + ".txt", 'w') as g:
                                     writer = csv.writer(g)
                                     for j in txt:
                                        g.write(j.encode('utf-8'))

                       except:
                           pass
