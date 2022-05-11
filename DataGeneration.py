# Data generation - facebook comments scraping from "important news sources from Mexico"

from http import cookies
from facebook_scraper import get_posts
import pandas as pd


comments = []
comments_Universal = []
comments_Milenio = []
comments_Excelsior = []
comments_Reforma = []
comments_Maerker = []
comments_Economista = []
comments_Azteca = []
comments_Aristegui = []
comments_Univision = []

#for post in get_posts('ElUniversalOnline', pages=25, cookies = "from_browser", options={"comments": True}):
#    comments_Universal = comments_Universal + [d['comment_text'] for d in  post['comments_full'][:100]]
    
#for post in get_posts('MilenioDiario', pages=25, cookies = "from_browser", options={"comments": True}):
#    comments_Milenio = comments_Milenio + [d['comment_text'] for d in  post['comments_full'][:100]]

#for post in get_posts('ExcelsiorMex', pages=25, cookies = "from_browser", options={"comments": True}):
#    comments_Excelsior = comments_Excelsior + [d['comment_text'] for d in  post['comments_full'][:100]]

#for post in get_posts('Reforma', pages=25, cookies = "from_browser", options={"comments": True}):
#    comments_Reforma = comments_Reforma + [d['comment_text'] for d in  post['comments_full'][:100]]
    
#for post in get_posts('denisemaerkeroficial', pages=25, cookies = "from_browser", options={"comments": True}):
#    comments_Maerker = comments_Maerker + [d['comment_text'] for d in  post['comments_full'][:100]]

#for post in get_posts('ElEconomista.mx', pages=100, cookies = "from_browser", options={"comments": True}):
#    comments_Economista = comments_Economista + [d['comment_text'] for d in  post['comments_full'][:50]]
    
for post in get_posts('AztecaNoticias', pages=100, cookies = "from_browser", options={"comments": True}):
    comments_Azteca = comments_Azteca + [d['comment_text'] for d in  post['comments_full'][:50]]
    
#for post in get_posts('AristeguiOnline', pages=100, cookies = "from_browser", options={"comments": True}):
#    comments_Aristegui = comments_Aristegui + [d['comment_text'] for d in  post['comments_full'][:50]]

#for post in get_posts('univisionnoticias', pages=100, cookies = "from_browser", options={"comments": True}):
#    comments_Univision = comments_Univision + [d['comment_text'] for d in  post['comments_full'][:50]]
    

#comments = comments_Economista

#print(len(comments_Economista))

#origin_E = ['Economista']*len(comments_Economista)

#df_comments_E = pd.DataFrame({'Comments': comments_Economista, 'Origin': origin_E})
#df_comments_E.to_csv('data/comments_wOriginEconomista.csv') 


print(len(comments_Azteca))

origin_Az = ['Azteca']*len(comments_Azteca)

df_comments_Az = pd.DataFrame({'Comments': comments_Azteca, 'Origin': origin_Az})
df_comments_Az.to_csv('data/comments_wOriginAzteca.csv')

 
#print(len(comments_Aristegui))

#origin_Ari = ['Aristegui']*len(comments_Aristegui)

#df_comments_Ari = pd.DataFrame({'Comments': comments_Aristegui, 'Origin': origin_Ari})
#df_comments_Ari.to_csv('data/comments_wOriginAristegui.csv') 


#print(len(comments_Univision))

#origin_U = ['Univision']*len(comments_Univision)

#df_comments_U = pd.DataFrame({'Comments': comments_Univision, 'Origin': origin_U})
#df_comments_U.to_csv('data/comments_wOriginUnivision.csv')