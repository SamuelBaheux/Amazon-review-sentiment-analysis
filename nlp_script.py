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


class Recommandation(Nlp):
    def __init__(self):
        super().__init__()
        self.dico_jeux_exemple = {'Cyberpunk' : ['B07DK1H3H5','B07SMBK245','B07TJ5Z389','B07S7RQST5'],
                     'The witcher 3' : ['B07Z9Z39ZW','B087T1FS9K','B00DE88BU6','B01FUCV5K6','B00L4SD1F8','B01L0TM25U','B07SZJQM7P','B0045Y2IVQ','B07J2MQZK5','B00FUC6SZO','B0858VSZCW','B0BRNZRPSH','B01L26C0US','B07X3ZWL7X','B00ICWO1XA','B07X6JP2QF','B00WRJCRP8','B074MY5TN6','B0BRNY8987','B01DDRA9UW','B00WV7PT9W','B01B4YOBDM'],
                     'Fifa22' : ['B089GGLKHQ','B08KJPPBQK','B08J27JV17','B08BMTDPLG','B08J2BPWGS','B08J2454R3','B08HWGBNLJ','B08BP3XFNH','B08B9Z6573','B08BMVN1PZ','B08J248GXK','B08NDRGH4K','B08J28WVHL','B08J2HDGKX','B08KTHWR1D','B08KXSFFJ6','B08GPQQTX1','B09486H473'],
                     'Call of Duty: Modern Warfare' : ['B07SNN8GV5','B074CB2RNQ','B07ZNVGJS1']
                     }
        self.categories_bonnes = {
            'graphismes': ['graphism', 'graphics', 'graphic', 'look', 'visuals', 'art', 'design', 'aesthetics',
                           'rendering', 'resolution', 'detail', 'texture', 'animation'],
            'gameplay': ['gameplay', 'combat', 'fight', 'experience', 'mechanics', 'controls', 'interaction',
                         'challenge', 'pace', 'flow', 'depth', 'strategy', 'tactics'],
            'enfants': ['kids', 'kid', 'son', 'daughter', 'sons', 'daughters', 'children', 'child', 'young ones',
                        'offspring', 'little ones', 'family'],
            'musiques_sons': ['headset', 'music', 'musics', 'sound', 'sounds', 'audio', 'ambient', 'score', 'tracks',
                              'effects', 'voiceover', 'dialogue'],
            'bon_scenario_histoire': ['story', 'characters', 'character', 'feel', 'feelings', 'feels', 'narrative',
                                      'plot', 'lore', 'world-building', 'dialogue', 'twists', 'development'],
            'se_joue_avec_manette': ['controller', 'controllers', 'gamepad', 'joystick', 'input device', 'remote',
                                     'pad', 'analog stick', 'd-pad', 'motion controls'],
            'open_world': ['open', 'liberty', 'freedom', 'sandbox', 'exploration', 'non-linear', 'vast', 'expansive',
                           'immersive', 'world design', 'environment', 'map size', 'boundless'],
            'jeu_facile': ['easy', 'accessible', 'simple', 'casual', 'beginner-friendly', 'user-friendly',
                           'approachable', 'undemanding', 'relaxed'],
            'jeu_faisant_partie_dune_saga': ['first', 'better', 'improved', 'sequel', 'prequel', 'series', 'franchise',
                                             'installment', 'continuation', 'universe', 'lore', 'canon'],
            'jeu_original': ['original', 'new', 'innovative', 'unique', 'fresh', 'creative', 'novel', 'groundbreaking',
                             'distinctive', 'trailblazing', 'avant-garde'],
            'rapport_qualite_prix': ['price', 'prices', 'worth', 'quality', 'works', 'buy', 'bought', 'value',
                                     'affordable', 'cost-effective', 'investment', 'budget-friendly', 'expensive'],
            'jeu_fun': ['fun', 'happy', 'enjoyable', 'entertaining', 'amusing', 'joyful', 'lighthearted', 'delightful',
                        'cheerful', 'playful', 'upbeat', 'spirited'],
            'jeu_sombre': ['sad', 'tears', 'dead', 'death', 'depressing', 'bleak', 'gloomy', 'tragic', 'grim', 'morbid',
                           'melancholic', 'dark'],
            'jeu_dur': ['hard', 'hardcore', 'difficult', 'though', 'challenging', 'punishing', 'demanding', 'intense',
                        'grueling', 'tough', 'unforgiving', 'steep']
        }
        self.categories_mauvaises = {
            'jeu_bug': ['bugs', 'bug', 'glitch', 'glitches', 'errors', 'crashes', 'issues', 'problems', 'flaws',
                        'defects'],
            'mauvais_gameplay': ['gameplay', 'combat', 'fight', 'experience', 'mechanics', 'controls', 'interaction',
                                 'challenge', 'pace', 'flow', 'depth', 'strategy', 'tactics'],
            'jeu_trop_volumineux': ['gb', 'GB', 'giga', 'gigas', 'go', 'GO', 'Gb', 'gig', 'Gig', 'download', 'Download',
                                    'update', 'install'],
            'jeu_trop_cher': ['price', 'prices', 'worth', 'quality', 'works', 'buy', 'bought', 'value', 'affordable',
                              'cost-effective', 'investment', 'budget-friendly', 'expensive', 'money', 'cost', 'costs'],
            'mauvais_graphismes': ['graphism', 'graphics', 'graphic', 'look', 'visuals', 'art', 'design', 'aesthetics',
                                   'rendering', 'resolution', 'detail', 'texture', 'animation']
        }

    def calculate_prop_adj_jeu(self, jeu, cat, sentiment):
        parent = self.dico_jeux_exemple[jeu]
        df_temp = self.df2[self.df2['parent_asin'].isin(parent)].copy()
        df_temp = df_temp[df_temp['sentiment_pred'] == sentiment]
        df_temp['text'] = df_temp['text'].fillna('')
        print(df_temp.shape)
        prop_adj = {}
        for key in cat.keys():
            part_key = round(df_temp['text'].apply(lambda x: sum(word in x for word in cat[key])).sum()/df_temp.shape[0]* 100, 2)
            prop_adj[key] = part_key
        return prop_adj




