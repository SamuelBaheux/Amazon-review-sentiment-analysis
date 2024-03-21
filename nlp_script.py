import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from nltk.sentiment import SentimentIntensityAnalyzer

nltk.download("punkt")
nltk.download("stopwords")
nltk.download('vader_lexicon')


class Dataprep:
    def __init__(self):
        self.df = None
        self.df2 = None
        self.ps = PorterStemmer()

    def import_data(self):
        self.df = pd.read_csv('./data/data_JV_2020_2023_tokenised.csv')

    def datapreprocess(self):
        self.df['title_y'] = self.df['title_y'].fillna('')
        self.df['text'] = self.df['text'].fillna('')
        self.df2 = self.df.iloc[:2000, :].copy()

    def stem_word(self, word):
        return self.ps.stem(word)

    def stem_column(self, column):
        return column.apply(lambda x: ' '.join([self.stem_word(word) for word in nltk.word_tokenize(x)]))

    def remove_stop_words(self, text):
        tokens = word_tokenize(text)
        stop_words = set(stopwords.words("english"))
        filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
        self.filtered_text = ' '.join(filtered_tokens)

    def supp_stopwords(self):
        self.df2["avis_stopwords"] = self.df2["text"].apply(self.remove_stop_words)
        self.df2["avis_preprocess"] = self.stem_column(self.df2["avis_stopwords"])


class Nlp(Dataprep):
    def __init__(self):
        super().__init__()
        self.MODEL_NAME = "LiYuan/amazon-review-sentiment-analysis"
        self.TOKENIZER = AutoTokenizer.from_pretrained(self.MODEL_NAME)
        self.MODEL = AutoModelForSequenceClassification.from_pretrained(self.MODEL_NAME, from_tf=False)
        self.sentiment_task = pipeline("sentiment-analysis", model=self.MODEL, tokenizer=self.TOKENIZER)
        self.result = None
        self.sia = SentimentIntensityAnalyzer()
        self.mots_frequents = None
        self.mots_freq_positifs = None
        self.mots_freq_negatifs = None

    def get_sentiment(self, text):
        max_length = 512
        text = text[:max_length]
        self.result = self.sentiment_task(text)[0]
        return self.result["label"]

    def apply_sentiment(self):
        self.df2['sentiment_HF'] = self.df2['text'].apply(self.get_sentiment)

    def conversion_en_int(self):
        self.df2['rating'] = self.df2['rating'].astype('int')
        self.df2['sentiment_pred'] = self.df2['sentiment_HF'].apply(
            lambda x: 1 if x == 1 else 1 if x == 2 else 2 if x == 3 else 3)
        self.df2['sentiment_true'] = self.df2['rating'].apply(
            lambda x: 1 if x == 1 else 1 if x == 2 else 2 if x == 3 else 3)
        self.df2['perf_count'] = self.df2.apply(lambda row: 1 if row['sentiment_pred'] == row['sentiment_true'] else 0,
                                                axis=1)
        print(
            f"Performance : {round(self.df2['perf_count'].sum() / self.df2.shape[0] * 100, 2)}% de bonnes predictions")

    def mots_impactants(self,commentaires):
        self.mots_frequents = []
        mots_a_exclure = ['br', 'game', 'great', 'like', 'playing', 'one', 'good', 'games', 'play', 'love', 'get',
                          'well', 'really', 'would', 'time', 'use',
                          'nice', 'also', 'nice', 'even', 'much', 'playing', 'case', 'got', 'loves', 'played']

        for commentaire in commentaires:
            if self.sia.polarity_scores(commentaire)['compound'] > 0:
                mots = [mot.lower() for mot in word_tokenize(commentaire) if
                        mot.isalpha() and mot.lower() not in stopwords.words(
                            'english') and mot.lower() not in mots_a_exclure]
                self.mots_frequents.extend(mots)

        return self.mots_frequents

    def recup_mots_positifs_negatifs(self):
        self.mots_freq_positifs = FreqDist(self.mots_impactants(self.df2[self.df2['sentiment_pred'] == 3]['text']))
        self.mots_freq_negatifs = FreqDist(self.mots_impactants(self.df2[self.df2['sentiment_pred'] == 1]['text']))


class graphics_and_plots(Nlp):
    def __init__(self):
        super().__init__()
        self.wordcloud = None

    def word_cloud_negatif(self):
        WordCloud(width=900, height=500, background_color='white').generate_from_frequencies(
            self.mots_freq_negatifs)
        plt.figure(figsize=(10, 5))
        plt.imshow(self.wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.show()

    def word_cloud_positif(self):
        WordCloud(width=900, height=500, background_color='white').generate_from_frequencies(
            self.mots_freq_positifs)
        plt.figure(figsize=(10, 5))
        plt.imshow(self.wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.show()


