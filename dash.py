import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from wordcloud import WordCloud,STOPWORDS
import pickle
import pydeck as pdk
import re
from collections import Counter
from PIL import Image
import datetime


#import variables

#########################  a faire #########################################
# 
#
###########################################################################"


#Variables Correl Description
#becomes 
#variable_x variable_y description


st.set_page_config(layout="wide")

#import des données
@st.cache
def load_data():
	data=pd.read_csv('tweets1.csv',sep='\t')
	data=data.append(pd.read_csv('tweets2.csv',sep='\t'))
	data['created_at']=data['created_at'].apply(lambda x:pd.to_datetime(x))
	return data

def draw_numbers_pos_neg(since,until,pos,neg,titre):
	p=pos.groupby('created_at').aggregate({'created_at':'count'})
	n=neg.groupby('created_at').aggregate({'created_at':'count'})
	legend,tickes=echelle(since,until)
	X=[k for k in daterange(since,until)]
	x=[str(k) for k in daterange(since,until)]
	y=[p.loc[i].values[0] if i in p.index else 0 for i in x]
	y2=[-n.loc[i].values[0] if i in n.index else 0 for i in x]
	
	
	fig = go.Figure(go.Bar(name='Optiistic tweets per day',x=X,y=y,marker_color='green'))
	fig.add_trace(go.Scatter(name='Week mean number', x=X,y=moyenne(y), mode='lines',marker_color='lightgreen'))
	fig.add_trace(go.Bar(name='Pessimistic tweets per day',x=X,y=y2,marker_color='indianred'))
	fig.add_trace(go.Scatter(name='Week mean number', x=X,y=moyenne(y2), mode='lines',marker_color='coral'))
	
	#fig.update_layout(autosize=True,legend=dict(orientation="h",yanchor="bottom",y=1.02,xanchor="center",x=0.5))
	#fig.update_xaxes(ticktext=legend,tickvals=tickes)
	fig.update_layout(title=titre,legend=dict(orientation="h",yanchor="bottom",y=1.02,xanchor="right", x=1,font=dict(size=15),title=dict(font=dict(size=15))))
	
	return fig
	
def draw_numbers(since,until,df,titre):
	d=df.groupby('created_at').aggregate({'created_at':'count'})
	
	legend,tickes=echelle(since,until)
	X=[k for k in daterange(since,until)]
	x=[str(k) for k in daterange(since,until)]
	y=[d.loc[i].values[0] if i in d.index else 0 for i in x]
	
	fig = go.Figure(go.Bar(name='Tweets per day',x=X,y=y))
	fig.add_trace(go.Scatter(name='Week mean number', x=X,y=moyenne(y), mode='lines'))
	
	#fig.update_layout(autosize=True,legend=dict(orientation="h",yanchor="bottom",y=1.02,xanchor="center",x=0.5))
	#fig.update_xaxes(ticktext=legend,tickvals=tickes)
	fig.update_layout(title=titre,legend=dict(orientation="h",yanchor="bottom",y=1.02,xanchor="right", x=1,font=dict(size=15),title=dict(font=dict(size=15))))
	
	return fig

def echelle(since,until):
	start=since.month
	end=until.day
	months=12*(until.year-since.year)+until.month-since.month
	year=since.year
	month=since.month
	legend=[calendar[start]+str(year)[-2:]]
	
	for i in range(months):
		legend.append('')
		month+=1
		if month>12:
			month=1
			year+=1
		legend.append(calendar[month]+str(year)[-2:])
	if end>15:
		legend.append('')
	
	ticks=['' for i in range(len(legend))]
	year=since.year
	month=start
	for i in range(len(ticks)//2):
		ticks[2*i]=datetime.date(year,month,15)
		if len(ticks)>2*i+1:
			ticks[2*i+1]=datetime.date(year,month,calendar2[legend[2*i][:-2]])
		ticks[-1]=datetime.date(until.year,until.month,until.day)
		month+=1
		if month>12:
			month=1
			year+=1
	#traiter les années bisextiles
	return legend,ticks

	
def moyenne(liste):
    if len(liste)<=14:
        return liste
    else:
        L=[0 for i in range(len(liste))]
        for i in range(7):
            j=i+1
            somme=sum([liste[i+k] for k in range(8)])
            somme2=sum([liste[-(j+k)] for k in range(8)])
            for k in range(i):
                somme+=liste[i-k-1]
                somme2+=liste[-j+k+1]
            L[i]=somme/(8+i)
            L[-j]=somme2/(8+i)
        for i in range(7,len(liste)-7):
            L[i]=sum([liste[i-7+k] for k in range(14)])/14
        return L
    
def daterange(date1, date2):
    for n in range(int ((date2 - date1).days)+1):
        yield date1 + datetime.timedelta(n)

def check_words(x,words):
	for word in words:
		if word in x:
			return True
	return False 

calendar={1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'Mai',6:'Jun',7:'Jul',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'}
calendar2={'Jan':31,'Feb':28,'Feb2':29,'Mar':31,'Apr':30,'Mai':31,'Jun':30,'Jul':31,'Aug':31,\
           'Sep':30,'Oct':31,'Nov':30,'Dec':31}

	

#st.write(data.columns)
#st.write(correl.shape)


def main():	
	
	data=load_data()
	
	topic = st.sidebar.radio('Select which tweets you want to visualize?',('All Tweets','Election','Security','Disaster'))
	
	if topic in ['Election','Security','Disaster']:
		if topic=='Security':
			theme='insecurity'
		else:
			theme=topic.lower()
		tweets=data[data[theme]>0.8]
		#st.title("Total: "+str(len(tweets))+" Tweets")
		
	
	else:
		#st.title("Total: "+str(len(data))+" Tweets")
		tweets=data.copy()
		
	positive=tweets['optimistic']>0.5
	
	col1,stop,col2 = st.columns((5,2,5)) # To make it narrower
	form = 'MMM DD, YYYY'  # format output
	start_date = datetime.date.fromisoformat('2020-01-01')  #  I need some range in the past
	end_date = datetime.date.fromisoformat('2021-07-30')
	end_date2 = datetime.date.fromisoformat('2021-07-31')
	
	debut = col1.slider('Select start', min_value=start_date, value=start_date ,max_value=end_date, format=form)
	fin = col2.slider('Select end', min_value=debut, value=end_date2 ,max_value=end_date2, format=form)
	               
	#st.dataframe(tweets)
	#st.dataframe(tweets[positive])
	#st.dataframe(tweets[-positive])
	
	fig= draw_numbers(pd.to_datetime("2020-01-01"),pd.to_datetime("2021-07-31"),tweets,'Tweets per day') 
	
	fig.update_layout(shapes=[dict(type="rect",xref="x",yref="y",x0=debut,y0="0",x1=fin,y1="400",
        fillcolor="lightgray", opacity=0.9,line_width=0,layer="below"),])
	
	fig2=draw_numbers_pos_neg(debut,fin,tweets[positive],tweets[-positive],'Tweets per day') 
	
	st.plotly_chart(fig,use_container_width=True)
	
	tweets=tweets[tweets['created_at']>=pd.to_datetime(debut)]
	tweets=tweets[tweets['created_at']<=pd.to_datetime(fin)]
	
	st.subheader('There are a total of '+str(len(tweets))+' tweets between the '+debut.strftime('%d %b %Y')+' and the '+fin.strftime('%d %b %Y'))
	
	st.plotly_chart(fig2,use_container_width=True)
	
		#st.write(questions)
		#st.write(cat_cols)
	
	col1,stop,col2 = st.columns((5,1,5)) 	
	
	x, y = np.ogrid[100:500, :600]
	mask = ((x - 300)/2) ** 2 + ((y - 300)/3) ** 2 > 100 ** 2
	mask = 255 * mask.astype(int)		
	
	sw=STOPWORDS
	sw.add('t')
	sw.add('https')
	sw.add('co')

	corpus=' '.join(tweets['text'])
	corpus=re.sub('[^A-Za-z ]',' ', corpus)
	corpus=re.sub('\s+',' ', corpus)
	corpus=corpus.lower()
	if corpus==' ' or corpus=='':
		corpus='Nothing to display'
	else:
		corpus=' '.join([i for i in corpus.split(' ') if i not in sw])
	wc = WordCloud(background_color="#0E1117", repeat=False, mask=mask)		
	wc.generate(corpus)
	col1.image(wc.to_array(), use_column_width = True)
	
	df=tweets[['created_at','text']].copy()
	df.sort_values(by='created_at',inplace=True)
	df.columns=['date','tweets']
	df['date']=df['date'].apply(lambda x:str(x)[:10])
	col2.dataframe(df)


    
 
if __name__== '__main__':
    main()




    
