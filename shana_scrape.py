import requests
import pandas as pd
import numpy as np
from tqdm import tqdm 
from bs4 import BeautifulSoup 
from newspaper import Article 
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

def scrap():
    page = 0
    scraped_data  = []
    
    while True:
        page += 1

        main_page_url = f"https://www.shana.ir/archive?pi={page}&ms=0" 

        html = requests.get(main_page_url).text
    
        soup = BeautifulSoup(html , features = 'lxml' )

        links = soup.find_all('h3')
        
        if len(links) == 0:
            break
        
        for link in tqdm(links) :
            page_url = 'http://www.shana.ir/'+ link.a['href']
            try :
                
                article = Article(page_url)
                article.download()
                article.parse()
                scraped_data.append( {'url': page_url ,'text':article.text , 'title':article.title} )
                    
            except :
                print (f"failed process page : {page_url}")    
            
    df = pd.DataFrame(scraped_data)
    df.to_csv('shana.csv')    
            

scrap()

corpus = pd.read_csv('shana.csv', index_col= 0)

# count of docs

print(f'cout of corpus data: {len(corpus)}')

print(f'columns of curpus data :{corpus.shape}')

print(f'columns of curpus data is :{corpus.columns}')

corpus.head(5)

# tf-idf corpus

# skip docs without text
docs = corpus.loc[corpus.text.isnull() == False]['text']

vectorizer = TfidfVectorizer()
tfidf_docs = vectorizer.fit_transform(docs)

# tf-idf query

query = 'نفت'

tfidf_query = vectorizer.transform([query])[0]

# similarities

cosines = []

for d in tqdm(tfidf_docs):
  cosines.append(float(cosine_similarity(d, tfidf_query)))
  
# sorting

k = 10
sorted_ids = np.argsort(cosines)

for i in range(k):
  cur_id = sorted_ids[-i-1]
  cur_doc = corpus.iloc[cur_id]
  print(cosines[cur_id], cur_doc['title'])