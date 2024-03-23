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
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from nltk.sentiment import SentimentIntensityAnalyzer

nltk.download("punkt")
nltk.download("stopwords")


class Dataprep:
    def __init__(self):
        self.col = None
        self.filtered_text = None
        self.df = None
        self.df2 = None
        self.ps = PorterStemmer()
        self.dico_jeux_exemple = {'The Witcher 2': ['B00ICWO1XA', 'B07X6JP2QF'],
                                  'The Witcher 3': ['B00DE88BU6', 'B01L0TM25U', 'B07SZJQM7P', 'B07J2MQZK5',
                                                    'B0858VSZCW', 'B0BRNZRPSH', 'B01L26C0US', 'B07X3ZWL7X',
                                                    'B074MY5TN6', 'B0BRNY8987', 'B01DDRA9UW', 'B01B4YOBDM'],
                                  'Minecraft': ['B07D62Y1KQ', 'B0B5SVBKTR', 'B07DG77KPX', 'B010KYDNDG', 'B00NXSRP72',
                                                'B08M78SGGD', 'B07D13QGXM', 'B0B1H3DY7L', 'B06XH297M5', 'B09DDZHZF6',
                                                'B0815VFZRL', 'B09TWFC6SK', 'B08FV3CTVN', 'B07NY1LGBV', 'B07NTR1KL4',
                                                'B08FXX1LS3', 'B07JCRY8WP', 'B07CB12726', 'B082H2SR4K', 'B0746XH9M9',
                                                'B082WJXLPN', 'B077TQWCFB', 'B00KVQYJR8', 'B074KSX7PJ', 'B01LWMKQWU',
                                                'B0169NZNKI', 'B09JTJFYBF', 'B01F04ZDO8', 'B00CKKIJ24', 'B00I6E6SH6',
                                                'B07JMHZMX1', 'B00DUV027M', 'B00K2S5PUK', 'B01LZVTM35', 'B014XCWXIW',
                                                'B07NZ8QZW5', 'B00NGBTFWY', 'B08FN6FBD6', 'B00O8RMBLC', 'B014XCWYOK',
                                                'B082J7S5L6', 'B09GWH4BWZ', 'B01BVX6B02', 'B00BU3ZLJQ', 'B07DVTSDQ7',
                                                'B07D2XKQJH', 'B07DVX7V1R', 'B07NYGMPFJ', 'B07D9SB7XW', 'B07JX4PP8G',
                                                'B082J3D5HN', 'B00D4B8LX0', 'B09CDGRP3Y', 'B08PW55J48', 'B0141736OO',
                                                'B00K2S5PUA', 'B01LYOVTQT', 'B07TNCX3WZ', 'B09K8FH99X', 'B08JTZFDYV',
                                                'B01M0J6X90', 'B00T5EGJMK', 'B0155WN6G2', 'B07BK9JZHW', 'B014XCWZA8',
                                                'B08KHRKWGR', 'B082S9KV7X', 'B07L41PGS9', 'B0764RLHGK', 'B014XCWYPO',
                                                'B0898B8Z8Y', 'B08JV7K25C', 'B07D131MS4', 'B09YZQFCD1', 'B07TKD64W5',
                                                'B01EA8STK0', 'B07THVC34Y', 'B074X1FCXP', 'B084P62R48', 'B0170TUPVC',
                                                'B07HZD7S92'],
                                  'Borderlands 2': ['B005FUP0JG', 'B07V7DJ98N', 'B00KWH5CF4', 'B00HWFCHFW',
                                                    'B07V2BTV7X', 'B00NB5PU88', 'B00F4KXLDO'],
                                  'Mass Effect 2': ['B002JTX7JQ', 'B004GWQNN6', 'B00485CRTK'],
                                  'Dark Souls II': ['B07RBMZRP3', 'B07RFZFNQT', 'B06ZXRZ7Y8', 'B0191J0RXA',
                                                    'B01C2E4DTA', 'B075F9JXSG', 'B00Z9THPWS', 'B00Q6DC96S',
                                                    'B00AK4QB22', 'B00F6YD26Y', 'B006YDPU48', 'B00IVHQ1L6',
                                                    'B00QH0OCTW', 'B0191J0SES', 'B00F6YD27I'],
                                  'Portal 2': ['B004GUTRYK', 'B004IEA4QE', 'B00AK8QS76'],
                                  'Batman: Arkham City': ['B07WWK48DV', 'B07WWJDWY3'],
                                  'Uncharted 3': ['B00M3D8CFM', 'B0055ARKZI'],
                                  'Dishonored 2': ['B00ZM5OXD8', 'B00ZM5P3UK', 'B01L82CERU', 'B01GW8ZC2O', 'B01016WNNE',
                                                   'B01M7SHTEO'],
                                  'Far Cry 3': ['B07F19PVY5', 'B07MDJJ9BP', 'B07F3X4S4V', 'B005OGKYVK'],
                                  'Diablo III': ['B0050SZC5U', 'B00KCCNMYW', 'B00BIXYIU6', 'B01FCLK7CC', 'B073DKXJWF',
                                                 'B01J4JYIVO', 'B012JLWOW4', 'B07GLDK7JF', 'B00IJRW6Q2', 'B07DX2VCKW',
                                                 'B00KAPFOG0', 'B012JMXCQK', 'B00KLXU8WG', 'B07W4LW4M9', 'B07DX32MSP'],
                                  'Grand Theft Auto V': ['B0086VPUHI', 'B07H3TF8L7', 'B00I8S3HRY', 'B09L2PD5X9',
                                                         'B00B200UV6', 'B00A7QPNK4', 'B00B6ZBGVK', 'B00FMHV5U0',
                                                         'B00GM5TPSA', 'B001ELJE1U', 'B09WZ4ZH5Z', 'B016ASWQ2A',
                                                         'B0772TWYJJ', 'B07TWM13SJ', 'B079VW1S5T', 'B018TCAN7U',
                                                         'B09X4GSGSC', 'B07XTSKC8N', 'B07WSFFMN8', 'B07TWN9TJT',
                                                         'B00G7LO6OC', 'B01NA6PWAW', 'B00KL4AROO', 'B01J4KJH14',
                                                         'B001JFTDLM', 'B07T93B35K', 'B07TPFH7BS', 'B07TTHKR9J',
                                                         'B07GL7QPFY', 'B00CY92XU0'],
                                  'Grand Theft Auto IV': ['B001D8Q5MA', 'B06XDXV3CW', 'B078PQ9PWK', 'B07GBTCMC7',
                                                          'B00BZ9LDWO', 'B00FYXF5VC', 'B00557ZMOW'],
                                  'Bioshock Infinite': ['B003ZDOFF0', 'B009PJ9L3Y', 'B00EIN28I2', 'B00NY7H0T0',
                                                        'B009SPZ11Q', 'B00NY7H5EA', 'B00HV0MNKW'],
                                  'Dragon Age: Inquisition': ['B00K0NV5J2'],
                                  'Watch Dogs': ['B07SW1PC5K', 'B01GT5YI66', 'B01GKFPFZS', 'B00BGHUS58', 'B00FPQFXGK',
                                                 'B07SMBJYFX', 'B00BI83EVU', 'B07T38K3ZR', 'B00BHV4P0M', 'B00DYAQHTQ',
                                                 'B07SR1FMMC', 'B00DYAQHNC', 'B0B1P7CYSC', 'B07CYRD2HS', 'B08J4R7KS1',
                                                 'B078XXZJM1', 'B00CX8VY4S', 'B00BQWTGIS', 'B07S8DFBLS', 'B00OKZ2Y5K',
                                                 'B00DOUJJ0U', 'B01AUDYZHA', 'B07SY53LFJ'],
                                  'Bloodborne': ['B07DNLYMV4', 'B00NOD0OTW', 'B01MS1GOAX', 'B018209X7U', 'B018GVR638',
                                                 'B07DQN7SKM'],
                                  'Metal Gear Solid V: The Phantom Pain': ['B00T7HJZNU'],
                                  'Undertale': ['B076GS6NSP', 'B07H84SY83', 'B07J3N7QQV', 'B077H5PCPL', 'B076X4H7BR',
                                                'B07FK8CZX6', 'B077H56F6P', 'B077H5Q1ML'],
                                  'Persona 5': ['B085PS9F4H', 'B01MYUCFBK', 'B0872XQSBK', 'B0BF9L5CK8', 'B07R14ZFRN',
                                                'B07B956T7C', 'B08FD5PSR9', 'B08Q6WW8TY', 'B0BFDZDTJ7', 'B0BFDZDTJ8',
                                                'B08QH3BSD5', 'B084HYLQWW', 'B0BF1W7J6T', 'B0BFHY5FCP', 'B0BHV78MJR',
                                                'B07G7H3Y6Z', 'B0BLPDGKGT', 'B084WS4TK7', 'B082BW2HG9'],
                                  'Divinity: Original Sin 2': ['B07R7XQZXK', 'B07YQLKDZX'],
                                  'Red Dead Redemption 2': ['B07DK14HJF', 'B01M6C7YE8', 'B01M65RD19', 'B09FXV4JQF',
                                                            'B07DNKZJB2', 'B0776YMQN1', 'B01M6Y1Y4A'],
                                  'Monster Hunter: World': ['B072MQNKYV', 'B075S1VZZ7', 'B07G7NDVRN'],
                                  'Sekiro: Shadows Die Twice': ['B08F1Y6WB2'],
                                  'The Outer Worlds': ['B07SPKJRWX', 'B07ZPF625Q', 'B07SR34FJ7', 'B07ZL89998',
                                                       'B084H5J2TP', 'B08JHDHSC7', 'B08ZGMP3NB'],
                                  'Ghost of Tsushima': ['B098KX2GTR', 'B08MBMWVMJ', 'B09LR5FK2T', 'B085LXXGN2',
                                                        'B08BSKT43L', 'B085JNHFWY', 'B086XMMWC7', 'B07D4XKRS4'],
                                  'Cyberpunk 2077': ['B07DK1H3H5', 'B07SMBK245', 'B07TJ5Z389', 'B07S7RQST5',
                                                     'B07S6MW77X', 'B07SF1LZ9Q'],
                                  'Hades': ['B0976QY95Y', 'B08WWC6GBX', 'B08X2K6B1Z'],
                                  'Elden Ring': ['B09N77RFYL', 'B0B271ZPZX', 'B07TK41TDV', 'B09R3HR7NY', 'B0B7RSBXFT',
                                                 'B09V1Y74G9', 'B09R3C1QHH'],
                                  'God of War Ragnarök': ['B0B6218TMM', 'B0B6LNYL8D', 'B0B61Y8SXC'],
                                  'The Legend of Zelda: Tears of the Kingdom': ['B0CBWJXHZF', 'B0BX7NC5K1',
                                                                                'B0BV9NKGJ4', 'B0BG9VFZ5L',
                                                                                'B0C44BZ31P', 'B0BVW3SJMF',
                                                                                'B0C61RX8G2'],
                                  'Hogwarts Legacy': ['B09VXBDJ9N', 'B0BT2WLW1X', 'B0C7R8VW3B', 'B09W4HF94W',
                                                      'B0B9YSZ9D9', 'B0B9YR7FSJ'],
                                  'Diablo IV': ['B0BT16CTX9']
                                  }

    def import_data(self):
        self.df = pd.read_csv('./data/data_JV_2020_2023_tokenised.csv')
        print(f'Shape initiale du dataframe : {self.df.shape}')
        values_list = [val for sublist in self.dico_jeux_exemple.values() for val in sublist]
        self.df = self.df[self.df['parent_asin'].isin(values_list)]
        print(f'Shape finale du dataframe : {self.df.shape}')
        print('import done')
        return self.df


    def datapreprocess(self):
        self.df['title_y'] = self.df['title_y'].fillna('')
        self.df['text'] = self.df['text'].fillna('')
        self.df2 = self.df[['text','title_x','title_y','parent_asin','rating']].copy()
        self.df2 = self.df2.iloc[:1500, :].copy()
        print('preprocess done')
        
    def stem_column(self, column):
        self.col = column.apply(lambda x: ' '.join([self.ps.stem(word) for word in nltk.word_tokenize(x)]))
        print('stem done')
        return self.col

    def remove_stop_words(self, text):
        tokens = word_tokenize(text)
        stop_words = set(stopwords.words("english"))
        filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
        return ' '.join(filtered_tokens)

    def supp_stopwords(self):
        self.df2["avis_stopwords"] = self.df2["text"].apply(lambda x: self.remove_stop_words(x))
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
        self.df2 = self.df2

    def get_sentiment(self, text):
        max_length = 512
        text = text[:max_length]
        self.result = self.sentiment_task(text)[0]
        return self.result["label"]

    def apply_sentiment(self):
        if self.df2 is not None:
            self.df2['text'] = self.df2['text'].fillna('')
            self.df2['sentiment_HF'] = self.df2['text'].apply(lambda x: self.get_sentiment(x))
            print('get sentiment applied')
        else:
            print("Erreur: Aucune donnée disponible dans df2.")
    def conversion_en_int(self):
        self.df2['sentiment_HF'] = self.df2['sentiment_HF'].apply(lambda x: x[0])
        self.df2['sentiment_HF'] = self.df2['sentiment_HF'].astype('int')
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
        print('positive negative words done')

class Graphics_and_plots(Nlp):
    def __init__(self):
        super().__init__()
        self.wordcloud = None

    def generate_wordcloud(self, mots_freq):
        return WordCloud(width=900, height=500, background_color='white').generate_from_frequencies(mots_freq)

    def word_cloud_positifs(self):
        if self.mots_freq_positifs is not None:
            self.wordcloud = self.generate_wordcloud(self.mots_freq_positifs)
            plt.figure(figsize=(10, 5))
            plt.imshow(self.wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.show()
        else:
            print("Erreur: Aucune donnée disponible pour les mots positifs.")

    def word_cloud_negatifs(self):
        if self.mots_freq_negatifs is not None:
            self.wordcloud = self.generate_wordcloud(self.mots_freq_negatifs)
            plt.figure(figsize=(10, 5))
            plt.imshow(self.wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.show()
        else:
            print("Erreur: Aucune donnée disponible pour les mots positifs.")



class Recommandation(Nlp):
    def __init__(self):
        super().__init__()
        self.prop_adj_bad = None
        self.prop_adj_good = None
        self.df_jeux_qualités = pd.DataFrame()

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
                        'grueling', 'tough', 'unforgiving', 'steep'],
            'aventure' : ['world','discover','exploration']
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

    def calculate_prop_mot_bon(self, jeu):
        parent = self.dico_jeux_exemple[jeu]
        df_temp = self.df2[self.df2['parent_asin'].isin(parent)].copy()
        df_temp = df_temp[df_temp['sentiment_pred'] == 3]
        df_temp['text'] = df_temp['text'].fillna('')
        self.prop_adj_good = {}
        for key in self.categories_bonnes.keys():
            part_key = df_temp['text'].apply(lambda x: sum(word in x for word in self.categories_bonnes[key])).sum()
            self.prop_adj_good[key] = part_key
        return self.prop_adj_good
    
    def calculate_prop_mot_mauvais(self, jeu):
        parent = self.dico_jeux_exemple[jeu]
        df_temp = self.df2[self.df2['parent_asin'].isin(parent)].copy()
        df_temp = df_temp[df_temp['sentiment_pred'] == 1]
        df_temp['text'] = df_temp['text'].fillna('')
        self.prop_adj_bad = {}
        for key in self.categories_mauvaises.keys():
            part_key = df_temp['text'].apply(lambda x: sum(word in x for word in self.categories_mauvaises[key])).sum()
            self.prop_adj_bad[key] = part_key
        return self.prop_adj_bad

    def points_forts_jeu(self):
        data = {}
        for jeu in self.dico_jeux_exemple.keys():
            meilleures_qualites = self.calculate_prop_mot_bon(jeu)
            meilleures_qualites = sorted(meilleures_qualites.items(), key=lambda x: x[1], reverse=True)
            meilleures_qualites = meilleures_qualites[:3]
            data[jeu] = meilleures_qualites

        self.df_jeux_qualités = pd.DataFrame(data)
        self.df_jeux_qualités.columns = self.dico_jeux_exemple.keys()
        print('df final done')





