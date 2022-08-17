import streamlit as st
import numpy as np
import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import bs4
from bs4 import BeautifulSoup as bs
import requests
from wordcloud import WordCloud
from matplotlib.pyplot import imread
from matplotlib import pyplot as plt




nltk.download('stopwords')
from nltk.stem.porter import PorterStemmer
import re
from nltk.corpus import stopwords
import warnings
warnings.filterwarnings('ignore')

ps = PorterStemmer()

def clean_text(text):
    text = re.sub('[^a-zA-Z]', " ", text)
    text = text.lower()
    text = text.split()
    text = [ps.stem(word) for word in text if word not in stopwords.words("english")]
    text = ' '.join(text)
    return text

st.title("AMAZON PRODUCT SENTIMENT ANALYSIS")
st.header("Instructions")
st.markdown("1.You have paste review page link")
st.markdown("2.please link of all review's page")

url = st.text_input("paste url Here")
reviewlist = []
submitted =st.button("submit")
if submitted:
    def get_soup(url):
        r = requests.get("https://www.amazon.in/boAt-Smartwatch-Multiple-Monitoring-Resistance/product-reviews/B096VF5YYF/ref=cm_cr_dp_d_show_all_btm?ie=UTF8&reviewerType=all_reviews")
        soup = bs(r.text, "html.parser")
        return soup

    def get_reviews(soup):
        reviews = soup.find_all('div', {'data-hook':'review'})
        try:
            for item in reviews:
                review = {
                'tiltle':item.find('a',{'data-hook':'review-title'}).text.strip(),
                'rating':  float(item.find('i', {'data-hook': 'review-star-rating'}).text.replace('out of 5 stars', '').strip()),
                'review': item.find('span', {'data-hook': 'review-body'}).text.strip(),
                }
                reviewlist.append(review)
        except:
            pass
        
    for x in range (1,10):
        soup = get_soup(f"https://www.amazon.in/boAt-Smartwatch-Multiple-Monitoring-Resistance/product-reviews/B096VF5YYF/ref=cm_cr_dp_d_show_all_btm?ie=UTF8&reviewerType=all_reviews")
        get_reviews(soup)
        if not soup.find('li', {'class': 'a-disabled a-last'}):
            pass
        else:
            break
        
data = pd.DataFrame(reviewlist)
st.download_button(
        label="Download data as CSV",
        data=data.to_csv().encode("utf-8"),
        file_name='data.csv',
        mime='text/csv',
)
dataset = st.file_uploader("Choose a File",type = ['csv'])
if dataset is not None:
    data1 = pd.read_csv(dataset)
    st.write('## Data set')
    st.dataframe(data1,3000,500)
                
if st.button('Click for Result'):
     text=data1.iloc[0]["review"]
     st.header('Uncleaned Text Sample')
     st.write(text)
     st.header('Cleaned Text Sample')
     text1 = clean_text(data1.iloc[0]["review"])
     st.write(text1)
     data1['Cleaned_Text']=data1['review'].apply(clean_text)
     st.header('Wordcloud')
     ip_rev_string = " ".join(data1['Cleaned_Text'])
     wordcloud_ip = WordCloud(background_color='black',width=1800,height=1400).generate(ip_rev_string)
     fig=plt.figure( figsize=(10, 5))
     plt.imshow(wordcloud_ip)
     plt.axis('off')
     st.pyplot(fig)
     sid = SentimentIntensityAnalyzer()
     data1["Vader_Score"] = data1["Cleaned_Text"].apply(lambda review:sid.polarity_scores(review))
     data1["Vader_Compound_Score"]  = data1['Vader_Score'].apply(lambda score_dict: score_dict['compound'])
     data1["Result"] = data1["Vader_Compound_Score"].apply(lambda c: 'positive' if c > 0 else ('negative' if c < 0 else 'neutral'))
     st.header('Sentiment Analysis')
     st.balloons()
     Courses = ['positive','negative','neutral']
     values = list(data1["Result"].value_counts())
     fig1 = plt.figure(figsize = (10, 5))
     plt.bar(Courses, values)
     st.pyplot(fig1)
     fig = plt.figure(figsize = (10, 5))
     plt.pie(data1["Result"].value_counts(), labels = data1["Result"].value_counts().index, autopct="%.0f%%")
     st.pyplot(fig)


