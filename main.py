import os
import csv
import logging
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.initializers import TruncatedNormal
from tensorflow.keras.backend import clear_session
from tensorflow.keras import Sequential, Model, Input
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D, LayerNormalization, MultiHeadAttention, Dropout, LSTM, GRU, Conv1D, MaxPooling1D, Flatten, BatchNormalization, Layer
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, roc_auc_score, precision_recall_curve, auc
from sklearn.model_selection import KFold
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler, SMOTE
from nltk.corpus import wordnet
from datetime import datetime
import argparse
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter, defaultdict
import nltk
from nltk.tokenize import word_tokenize
#nltk.download('punkt_tab')
csv.field_size_limit(100000000)
# Set up logging at the module level
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Mixed Precision Training
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# Define stopwords (could be loaded from an external source)
stopwords = ['awfully', 'nobody', "who's", 'latterly', 'ltd', 'thorough', 'liked', 'tq', 'bn', 'cx', 'less', 'sixty', 'sz', 'lj', 'which', 'substantially', 'ga', 've', 'bill', 'eighty', 'un', 'help', 'wants', 'es', 'whim', 'br', 'contains', 'too', 'merely', 'side', 'most', 'anybody', 'cg', 'anything', 'dx', 'inc', 'insofar', 'm2', 'think', 'don', 'p2', "ain't", "c'mon", 'somehow', 'com', 'dj', 'approximately', 'made', 'en', 'related', 'http', 'seems', 'whereafter', 't', 'furthermore', 'away', 'seven', 'several', "aren't", 'bc', 'ke', 'added', 'towards', 'usefulness', 'shows', 'part', 'vols', 'vt', 'brief', 'okay', 'wont', 'anyway', 'ab', 'hadn', "she'll", 'shes', 'meantime', 'seeing', 'ie', 'known', 'four', 'sincere', 'slightly', 'ev', 'he', 're', 'while', 'lo', 't1', 'unto', 'one', 'youre', 'vj', 'al', 'n', 'pp', 'probably', 'going', 'theres', 'fill', 'ep', 'gone', 'nl', 'dr', 'yours', 'n2', 'os', 'briefly', 'f', 'x3', 'past', '3b', 'therein', 'without', 'fl', 'second', 'become', 'hr', 'ending', 'her', 'else', 'tb', 'tc', 'otherwise', 'welcome', 'plus', 'better', 'anyways', 'ap', 'asking', 'no', 'oa', "wasn't", 'information', 'million', 'quickly', 'us', 'usefully', 'since', 'need', 'saw', 'ra', 'might', 'out', 'being', 'poorly', 'having', 'bottom', 'non', 'though', 'te', 'full', 'od', 'si', 'begin', 'fj', "c's", 'inner', 'therere', "here's", 'theyre', 'am', 'wheres', 'adj', 'mrs', 'thanx', 'va', 'ny', 'ourselves', 'and', 'section', "where's", 'although', 'either', 'reasonably', 'xo', 'right', 'sm', 'sn', 'km', 'everywhere', 'giving', 'novel', 'ou', 'ar', 'sure', 'obviously', 'sec', 'move', 'ok', 'sy', 'tries', 'shown', 'well-b', 'b3', 'eleven', 'biol', 'noone', 'research-articl', 'wouldn', 'indicates', 'rf', 'et-al', 'c2', 'taken', 'dc', 'therefore', 'until', 'ce', 'proud', 'omitted', '6o', 'ran', 'exactly', 'ei', 'is', 'using', 'very', 'fifth', 'must', 'f2', 'little', 'where', 'the', 'nevertheless', 'sq', 'oh', 'vo', 'words', 'widely', 'itd', 'a4', 'oc', 'rl', 'xi', "won't", 'world', 'au', 'throughout', 'ea', 'theyd', 'anywhere', 'isn', 'p', 'amongst', 'described', 'research', 'showns', 'bp', 'new', 'bu', 'following', 'each', 'haven', 'neither', 'ru', 'nn', 'suggest', "who'll", 'ia', 'kj', 'vq', 'cu', 'c1', 'know', 'jt', "haven't", "you're", 'present', 'herself', 'iq', 'various', 'far', 'try', 'mainly', 'fu', 'gives', 'fa', 'em', 'nor', 'aj', 'b', 'immediate', 'against', 'tried', 'home', 'ts', 'nearly', 'tt', 'ds', 'onto', 'they', 'oq', 'possible', 'respectively', 'becomes', 'ur', 'of', 'viz', 'normally', 'ip', 'given', 'd2', 'maybe', 'bx', 'ig', 'x2', 'ob', 'entirely', 'results', 'est', 'wherein', 'ec', "they'd", 'zero', 'ought', 'shed', 'three', 'herein', 'io', "we'll", 'according', 'needn', 'se', 'er', 'best', 'uj', 'dt', 'whats', 'two', 's2', 'anyhow', 'below', 'due', 'just', 'line', 'd', 'often', 'we', 'me', 'u201d', 'around', 'pq', '3a', "she'd", 'rh', 'similarly', 'upon', 'yet', 'course', 'accordance', 'nc', 'old', 'few', 'ju', 'owing', 'call', 'wa', 'all', "it'll", 'actually', 'at', 'hasn', 'fifteen', 'when', 'hid', 'abst', 'its', 'howbeit', 'tn', 'contain', "she's", 'la', 'i4', 'ct', 'associated', 'ag', 'immediately', 'recently', 'a', 'ex', 'by', 'gy', 'na', 'pt', 'greetings', 'saying', 'well', 'considering', 'rather', 'thats', 'sr', 'nd', 'cv', 'strongly', 'thereof', 'willing', 'unfortunately', 'specified', 'because', 'be', 'formerly', 'significantly', 'cn', 'uk', 'resulted', 'tell', "i've", 'qu', 'showed', 'ci', 'currently', 'cit', "mightn't", 'others', 'consequently', 'particular', 'pl', 'recent', 'gl', 'rr', "they've", 'uo', "we'd", 'z', 'believe', 'pf', 'definitely', 'ri', "they'll", 'followed', 'cj', 'fr', 'pagecount', 'twelve', 'mine', 'who', 'date', 'u', 'yr', 'anyone', "that've", 'ignored', 'containing', 'fs', 'affecting', "when's", 'bs', "that's", 'b2', 'obtain', 'rm', 'as', 'ask', '6b', 'changes', 'forth', "doesn't", 'available', 'usually', "we're", 'hes', 'way', "what'll", 'later', 'sent', "he'll", 'arise', 'accordingly', 'etc', 'go', 'nonetheless', 'indicated', 'effect', 'again', 'able', 'mn', 'my', 'p1', 'cc', 'relatively', 'find', 'near', 'interest', "weren't", 't3', 'trying', 'however', 'someone', 'apart', 'somewhat', 'regards', 'likely', 'besides', 'regarding', 'whos', 'or', 'ox', "wouldn't", 'myself', 'par', 'ib', 'somewhere', 'yourselves', 'cz', "didn't", 'gotten', 'thousand', 'something', 'le', 'that', 'pu', 'example', 'ih', 'almost', 'importance', 'found', 'it', 'largely', 'really', 'sf', 'unlikely', 'wouldnt', 'own', 'thoughh', 'ne', 'across', 'hence', 'shan', 'ups', 'cp', '0o', 'c3', 'xt', 'doesn', 'ni', 'whoever', 'xs', 'wonder', 'even', 'iy', 'there', 'him', 'got', 'di', 'run', 'serious', 'rj', 'six', 'them', 'ay', 'mt', 'nr', 'previously', 'moreover', 'fire', 'outside', "should've", 'itself', 'what', 'throug', 'about', 'always', 'i6', 'let', 'thereto', 'specify', 'ls', "hadn't", 'sc', 'td', 'already', 'gave', 'concerning', "i'd", 'keep', 'po', 'promptly', 'resulting', 'amount', 'ho', 'take', 'los', 's', 'thin', 'for', 'between', 'says', 'mean', 'becoming', 'describe', 'xj', 'make', 'see', 'nos', 'amoungst', 'av', 'aw', 'c', 'sl', 'aren', 'rs', 'dp', 'cd', 'dy', 'particularly', 'begins', 'behind', 'yourself', 'thereby', 'tl', 'was', 'x1', 'further', 'kg', 'presumably', 'causes', 'our', "shan't", 'hs', 'ry', 'same', 'ca', "isn't", 'put', 'looks', 'bk', 'afterwards', 'so', 'e3', 'uses', 'can', 'inward', 'cm', 'a3', 'i2', '0s', 'thanks', 'allows', 'were', 'gets', 'significant', "there's", 'h3', 'had', 'vd', 'this', 'whod', 'together', 'co', 'couldn', 'hers', "what's", 'sd', "t's", 'despite', 'rn', 'ey', 'ge', 'are', "don't", 'ml', 'an', 'mustn', 'sup', 'then', 'ed', 'pi', 'cant', 'everybody', 'may', 'ps', 'affected', 'um', 'goes', 'beside', 'bj', "needn't", 'fify', 'getting', 'possibly', 'i3', 'ue', 'lc', 'q', 'js', 'sometimes', 'edu', 'cr', 'ut', 'pk', 'used', 'system', 'o', 'thank', 'instead', 'm', 'predominantly', 'became', 'won', 'ones', 'alone', 'forty', "it's", 'thou', 'nj', 'successfully', 'mo', 'nay', 'on', 'whether', 'enough', 'readily', 'lr', 'tip', 'ft', 'consider', 'indicate', 'much', 'after', 'page', 'hereafter', 'indeed', 'like', 'say', 'ij', 'j', 'yt', "why's", 'df', 'hello', 'whomever', 'sometime', 'aside', 'e', 'eg', 'ss', 'here', 'ours', 'last', 'lest', 'necessary', 'wherever', 'will', 'www', 'fn', 'come', 'announce', 'twice', 'ng', 'oo', 'id', 'nine', 'pr', 'qj', 'you', 'sufficiently', 'ir', 'cf', 'a2', 'cy', 'many', 'seeming', 'overall', 'to', "can't", 'does', 'front', 'seem', 'cs', 'bt', 'useful', 'act', "couldn't", 'didn', 'please', 'ax', 'y2', 'gr', 'shouldn', 'ui', 'yj', 'inasmuch', 'iv', 'different', 'ao', 'oz', 'beyond', 'hed', 'iz', 'certain', 'k', 'before', 'll', 'pc', 'gi', 'been', 'self', 'a1', 'down', 'fc', 'tr', 'dd', 'xn', "i'll", 'mostly', 'somethan', 'y', 'gj', 'h2', 'ain', 'name', 'pm', 'et', 'invention', 'thence', 'unless', 'than', 'ol', 'wi', 'ln', 'except', 'ii', 'elsewhere', 'appear', 'r2', 'nothing', 'wasnt', 'next', 'namely', 'keeps', 'taking', 'til', 'tf', 'said', "i'm", 'vs', 'ns', 'appropriate', 'other', 'hundred', 'rd', 'somebody', 'xf', 'miss', 'st', 'among', 'thoroughly', 'x', 'not', 'under', 'every', 'former', 'tx', 'whenever', 'mu', "you'd", 'end', 'himself', "how's", "you've", 'how', 'ninety', 'heres', 'rt', 'wasn', 'during', 'per', 'detail', 'couldnt', 'ad', 'your', 'took', 'jj', 'but', 'wish', 'yl', 'nowhere', "hasn't", 'in', 'p3', 'bi', 'do', 't2', 'could', 'ae', 'into', 'ever', 'l', 'ot', 'selves', 'ti', 'ph', 'should', 'dk', 'fo', 'themselves', 'index', 'whom', 'cannot', 'apparently', 'ic', 'ow', 'refs', 'wo', 'ys', 'les', 'thickv', 'she', 'hopefully', 'stop', 'con', 'appreciate', 'came', 'cause', 'potentially', 'get', 'auth', 'such', 'fy', 'hereupon', 'que', 'whence', 'lt', 'similar', 'ff', "it'd", 'thereafter', 'whereas', 'within', 'ee', 'mg', 'ms', 'oj', 'da', 'eu', 'hereby', 'provides', 'least', 'pj', 'fi', 'truly', 'above', 'first', 'look', 'ac', 'xx', 'shall', 'once', 'el', 'eq', 'arent', 'from', 'back', 'lately', 'weren', 'b1', 'up', 'twenty', 'eight', 'follows', 'looking', 'over', 'noted', 'show', 'unlike', 'meanwhile', 'use', 'ma', "mustn't", 'specifying', 'only', 'ej', 'comes', 'soon', 'want', 'would', 'thered', 'did', 'specifically', 'dl', 'rc', 'ibid', 'secondly', 'thereupon', "they're", 'with', 'if', 'through', 'placed', 'toward', 'tm', 'have', 'nt', 'along', 'thru', "let's", 'seriously', 'cq', 'w', 'ch', 'corresponding', 'mug', 'seemed', 'sensible', 'some', 'has', 'volumtype', '3d', 'everything', 'du', 'i', 'pn', 'affects', 'both', 'done', 'primarily', 'now', 'quite', 'sa', 'whose', 'ref', 'pas', 'clearly', 'ko', 'ro', 'op', 'tends', 'doing', 'lf', 'important', 'e2', 'zi', 'another', 'anymore', 'more', 'these', "there'll", 'hardly', 'those', 'give', 'th', 'py', 'still', 'i7', 'mr', 'af', 'oi', 'mill', 'ten', 'cry', 'perhaps', 'empty', 'r', 'pd', 'via', 'off', 'ef', "you'll", 'yes', 'happens', 'pages', 'l2', 'qv', 'whole', 'xv', 'everyone', 'needs', 'bl', 'downwards', 'theirs', 'tj', "he'd", 'ix', 'regardless', "we've", 'youd', 'ord', 'beginning', 'eo', 'their', 'thus', 'vu', 'wed', 'went', 'il', 'latter', 'necessarily', 'means', 'also', 'hh', 'g', 'om', 'top', 'az', 'whatever', 'certainly', 'obtained', 'hi', 'im', 'hy', 'jr', 'i8', 'hither', 'v', 'vol', 'sp', 'sj', 'fix', 'whereupon', 'whereby', 'why', 'sorry', 'bd', 'pe', 'whither', "a's", 'never', 'beforehand', 'beginnings', 'og', 'rv', 'value', 'werent', 'xk', 'lets', 'xl', 'h', 'gs', 'none', 'any', 'rq', 'de', 'hu', 'lb', 'allow', 'especially', 'his', 'five', "he's", 'zz', 'hj', 'makes', 'ba', 'mightn', 'seen', 'tv', 'hasnt', "shouldn't", 'knows', "there've", 'third', 'tp', 'sub', 'ah', 'cl', "that'll", 'kept']  # Retain the original stopwords list

def current_time():
    """Returns the current date and time."""
    return datetime.now()

def print_timestamp(msg):
    """Prints the message with the current timestamp."""
    logging.info(f"{msg} date and time: {current_time()}")
    
def get_wordnet_pos(treebank_tag):
    """Convert treebank POS tags to WordNet POS tags"""
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN
    
def preprocess_text(sentence, remove_stopwords = True):
    """Preprocesses the text by converting to lowercase and optionally removing stopwords, applying stemming, and lemmatization."""
    sentence = sentence.lower()
    words = sentence.split()

    if remove_stopwords:
        from nltk.stem import WordNetLemmatizer
        from nltk import pos_tag
        lemmatizer = WordNetLemmatizer()
        pos_tags = pos_tag(words)  # Get POS tags for words
        words = [lemmatizer.lemmatize(w, get_wordnet_pos(pos)) for w, pos in pos_tags]
        words = [w for w in words if w not in stopwords]

    return ' '.join(words)

def load_data(file_path, remove_stopwords, target_label, test_size=None):
    """Loads and preprocesses the entire dataset from the CSV file and limits the dataset size based on data_size, preserving class proportions."""
    texts, labels = [], []

    try:        
        if test_size:
            print('Use small data')
            with open(file_path) as csvfile:
                reader = csv.reader(csvfile, delimiter=',')
                next(reader)  # Skip header

                for row in reader:
                    texts.append(row[2])
                    labels.append('1' if row[0] == target_label else '0')

            # Calculate the ratio of 0 and 1 in labels
            total_labels = len(labels)
            num_1 = labels.count('1')
            num_0 = labels.count('0')

            if total_labels > 0:
                ratio_1 = num_1 / total_labels
                ratio_0 = num_0 / total_labels
            else:
                ratio_1 = ratio_0 = 0

            print(f"Total labels: {total_labels}")
            print(f"Count of '1': {num_1} ({ratio_1:.2%})")
            print(f"Count of '0': {num_0} ({ratio_0:.2%})")

            num_samples_1 = int(test_size * ratio_1)
            num_samples_0 = test_size - num_samples_1 
            print(num_samples_1)
            print(num_samples_0)

            indices_1 = [i for i, label in enumerate(labels) if label == '1']
            indices_0 = [i for i, label in enumerate(labels) if label == '0']

            sampled_indices_1 = np.random.choice(indices_1, min(len(indices_1), num_samples_1), replace=False)
            sampled_indices_0 = np.random.choice(indices_0, min(len(indices_0), num_samples_0), replace=False)

            sampled_indices = np.concatenate([sampled_indices_1, sampled_indices_0])

            np.random.shuffle(sampled_indices)

            sampled_texts = [texts[i] for i in sampled_indices]
            texts = [preprocess_text(sentence, remove_stopwords=remove_stopwords) for sentence in sampled_texts]
            labels = [labels[i] for i in sampled_indices]

        else:
            print('Use full data')
            with open(file_path) as csvfile:
                reader = csv.reader(csvfile, delimiter=',')
                next(reader)  # Skip header

                for row in reader:
                    sentence = preprocess_text(row[2], remove_stopwords)
                    texts.append(sentence)
                    labels.append('1' if row[0] == target_label else '0')


    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise

    return texts, labels

def write_csv(project_name, header, data, apply_stopwords, model_type, task):
    """Writes the results to a CSV file."""
    file_path = f"./TEST/{model_type}/{project_name}/{task}_{'Apply' if apply_stopwords else 'Not_Apply'}_Stopword.csv"
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    try:
        with open(file_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(data)
    except Exception as e:
        logging.error(f"Error writing CSV file: {e}")
        raise

class PositionalEncoding(Layer):
    def __init__(self, maxlen, embedding_dim):
        super(PositionalEncoding, self).__init__()
        self.pos_encoding = self.positional_encoding(maxlen, embedding_dim)

    def positional_encoding(self, maxlen, embedding_dim):
        pos = np.arange(maxlen)[:, np.newaxis]
        i = np.arange(embedding_dim)[np.newaxis, :]
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(embedding_dim))
        pos_encoding = pos * angle_rates
        pos_encoding[:, 0::2] = np.sin(pos_encoding[:, 0::2])
        pos_encoding[:, 1::2] = np.cos(pos_encoding[:, 1::2])
        pos_encoding = pos_encoding[np.newaxis, ...]
        return tf.cast(pos_encoding, dtype=tf.float32)

    def call(self, x):
        x = tf.cast(x, dtype=tf.float32)
        seq_len = tf.shape(x)[1]
        return x + self.pos_encoding[:, :seq_len, :]

class PositionalEncoding_bert(tf.keras.layers.Layer):
    def __init__(self, maxlen, embed_dim, **kwargs):
        super(PositionalEncoding_bert, self).__init__(**kwargs)  
        self.maxlen = maxlen
        self.embed_dim = embed_dim
        self.positional_embeddings = self.add_weight(
            shape=(maxlen, embed_dim),
            initializer=TruncatedNormal(stddev=0.02),  
            trainable=False,  
            name='positional_embeddings'
        )

    def call(self, inputs):
        return inputs + self.positional_embeddings[:tf.shape(inputs)[1], :]
        
def build_model(model_type, vocab_size, embedding_dim, maxlen, filters=64, units=64, dropout_rate=0.1):
    """Builds and compiles the model based on the specified type."""
    clear_session()
    tf.random.set_seed(123)

    try:
        if model_type == "CNN":
            model = Sequential([
                Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=maxlen),
                Conv1D(filters=filters, kernel_size=3, activation='relu', padding='same'),
                BatchNormalization(),
                MaxPooling1D(pool_size=2),
                Flatten(),
                Dense(units=units, activation='relu'),
                Dropout(rate=dropout_rate),
                Dense(units=1, activation='sigmoid', dtype='float32')
            ])
        elif model_type == "LSTM":
            model = Sequential([
                Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=maxlen),
                LSTM(units=units, return_sequences=True),
                BatchNormalization(),
                LSTM(units=units),
                Dense(units=units, activation='relu'),
                Dropout(rate=dropout_rate),
                Dense(units=1, activation='sigmoid', dtype='float32')
            ])
        elif model_type == "GRU":
            model = Sequential([
                Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=maxlen),
                GRU(units=units, return_sequences=True),
                BatchNormalization(),
                GRU(units=units),
                Dense(units=units, activation='relu'),
                Dropout(rate=dropout_rate),
                Dense(units=1, activation='sigmoid', dtype='float32')
            ])
        elif model_type == "Transformer":
            inputs = Input(shape=(maxlen,))
            x = Embedding(input_dim=vocab_size, output_dim=embedding_dim)(inputs)

            x = PositionalEncoding(maxlen, embedding_dim)(x)

            for _ in range(4):
                attn_output = MultiHeadAttention(num_heads=4, key_dim=embedding_dim)(x, x)
                attn_output = tf.cast(attn_output, dtype=tf.float32)

                attn_output = Dropout(rate=dropout_rate)(attn_output)
                attn_output = tf.cast(attn_output, dtype=tf.float32)

                x = tf.cast(x, dtype=tf.float32)
                x = LayerNormalization(epsilon=1e-6)(x + attn_output)
                x = tf.cast(x, dtype=tf.float32)

                ffn_output = Dense(128, activation='relu')(x)
                ffn_output = Dense(embedding_dim)(ffn_output)
                ffn_output = tf.cast(ffn_output, dtype=tf.float32)
                x = LayerNormalization(epsilon=1e-6)(x + ffn_output)
                x = tf.cast(x, dtype=tf.float32)

                x = Dropout(rate=dropout_rate)(x)
                x = tf.cast(x, dtype=tf.float32)

            x = GlobalAveragePooling1D()(x)
            x = Dense(units, activation='relu')(x)
            x = BatchNormalization()(x)
            x = Dropout(dropout_rate)(x)

            outputs = Dense(1, activation='sigmoid', dtype='float32')(x)

            model = Model(inputs=inputs, outputs=outputs)

        elif model_type == "Bert":
            inputs = Input(shape=(maxlen,))
            x = Embedding(input_dim=vocab_size, output_dim=embedding_dim, embeddings_initializer=TruncatedNormal(stddev=0.02))(inputs)
            x = PositionalEncoding_bert(maxlen, embedding_dim)(x)
    
            for _ in range(4): 
                attn_output = MultiHeadAttention(num_heads=4, key_dim=embedding_dim)(x, x)
                attn_output = Dropout(rate=dropout_rate)(attn_output)
                x = LayerNormalization(epsilon=1e-6)(x + attn_output)

                ffn_output = Dense(128, activation='relu')(x)
                ffn_output = Dense(embedding_dim)(ffn_output)
                ffn_output = Dropout(rate=dropout_rate)(ffn_output)

                x = LayerNormalization(epsilon=1e-6)(x + ffn_output)

            x = GlobalAveragePooling1D()(x)
            x = Dense(units, activation='relu')(x)
            x = BatchNormalization()(x)
            x = Dropout(dropout_rate)(x)
            outputs = Dense(1, activation='sigmoid', dtype='float32')(x)
            model = Model(inputs=inputs, outputs=outputs)
            
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        logging.debug(f"Model {model_type} built successfully.")
    except Exception as e:
        logging.error(f"Error building model: {e}")
        raise

    return model

def calculate_metrics(y_true, y_pred_proba):
    """Calculates and returns evaluation metrics."""
    try:
        y_pred = (y_pred_proba > 0.5).astype(int)
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, average='macro'),
            "recall": recall_score(y_true, y_pred, average='macro'),
            "f1": f1_score(y_true, y_pred, average='macro'),
        }

        if len(np.unique(y_true)) < 2:
            fpr, tpr, thresholds = [np.nan], [np.nan], [np.nan]
            roc_auc = np.nan
        else:
            fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
            roc_auc = roc_auc_score(y_true, y_pred_proba)
        
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        PR_auc = auc(recall, precision)

        logging.debug(f"Metrics calculated successfully.")
    except Exception as e:
        logging.error(f"Error calculating metrics: {e}")
        raise

    return metrics, fpr, tpr, thresholds, roc_auc, PR_auc

def average_roc_curve(all_fpr, all_tpr):
    """Calculates the average ROC curve."""
    mean_fpr = np.linspace(0, 1, 100)
    mean_tpr = np.zeros_like(mean_fpr)
    try:
        for fpr, tpr in zip(all_fpr, all_tpr):
            mean_tpr += np.interp(mean_fpr, fpr, tpr)
        mean_tpr /= len(all_fpr)
        logging.debug("Average ROC curve calculated successfully.")
    except Exception as e:
        logging.error(f"Error calculating average ROC curve: {e}")
        raise

    return mean_fpr, mean_tpr

def parse_arguments():
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(description="BBC News Classification")
    parser.add_argument('project_name', type=str, help="Project name")
    parser.add_argument('task_name', type=str, help="Task name")
    parser.add_argument('target_label', type=str, help="Target label for classification")
    parser.add_argument('model_type', type=str, choices=['CNN', 'LSTM', 'GRU', 'Transformer', 'Bert'], help="Model type to use")
    parser.add_argument('data_type', type=str, help="Type of data to use (e.g., SUM+DES, other)")
    parser.add_argument('--num_words', type=int, default=1000, help="Number of words to consider in tokenization")
    parser.add_argument('--maxlen', type=int, default=120, help="Maximum length of sequences")
    parser.add_argument('--stopword_flag', action='store_true', help="Whether to apply stopword removal or not")
    parser.add_argument('--test_size', type=int, default=None, help="Size of the test set")
    parser.add_argument('--epochs', type=int, default=1, help="Number of epochs for training")
    parser.add_argument('--dropout_rate', type=float, default=0.1, help="Dropout rate for the model")
    parser.add_argument('--batch_size', type=int, default=256, help="Batch size for training")  # Increased for better GPU utilization
    parser.add_argument('--filters', type=int, default=64, help="Number of filters for CNN")
    parser.add_argument('--units', type=int, default=64, help="Number of units for LSTM/GRU/Transformer")
    parser.add_argument('--log_level', type=str, choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], default='INFO', help="Set the logging level")
    parser.add_argument('--use_smote', action='store_true', help="Use SMOTE for balancing the dataset")
    parser.add_argument('--num_selected_features', type=int, default=100, help="Number of top features to select")
    return parser.parse_args()

def create_tf_dataset(X, y, batch_size):
    """Creates a tf.data.Dataset for efficient data loading."""
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    dataset = dataset.shuffle(buffer_size=1024).batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset

def main():
    args = parse_arguments()
    logging.getLogger().setLevel(args.log_level)

    print_timestamp("Begin")

    file_suffix = f"{args.task_name}_{args.data_type}"
    data_file_path = f'./DATA/{args.project_name}/{args.project_name}_{file_suffix}.csv'
    
    # Load the entire dataset
    texts, labels = load_data(data_file_path, args.stopword_flag, args.target_label, args.test_size)
    
    # Initialize and fit tokenizer once on the entire dataset
    vectorizer = TfidfVectorizer(max_features=5000)  
    X_tfidf = vectorizer.fit_transform(texts).toarray()  
    
    y = np.array(labels).astype(int)
    num_selected_features = args.num_selected_features
    selector = SelectKBest(chi2, k=num_selected_features)
    X_selected = selector.fit_transform(X_tfidf, y)
    
    scaler = StandardScaler(with_mean=False)
    X = scaler.fit_transform(X_selected)
    
    feature_names = vectorizer.get_feature_names_out()

    selected_features_indices  = selector.get_support(indices=True)
    selected_feature_names = [feature_names[i] for i in selected_features_indices]
    
    combinations = []
    for _txt in texts:
        combined = f"{_txt} " + " ".join(selected_feature_names)
        combinations.append(combined)
    texts = combinations
    
    tokens = [word_tokenize(text.lower()) for text in texts]
    all_tokens = [token for sublist in tokens for token in sublist]
    vocab = Counter(all_tokens)

    vocab_size = len(vocab)
    print(f"Vocabulary size: {vocab_size}")
    
    tokenized_texts = [word_tokenize(text.lower()) for text in texts]
    lengths = [len(text) for text in tokenized_texts]

    maxlen = int(np.percentile(lengths, 95))
    print(f"95th percentile length: {maxlen}")
    
    tokenizer = Tokenizer(num_words=vocab_size)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    X = pad_sequences(sequences, maxlen)

    print('****************')
    if args.use_smote:
        print('SMOTE')
        smote = SMOTE(random_state=42)
        X, y = smote.fit_resample(X, y)
        print('*****************')
    else:
        print('RandomOverSampler')
        oversampler = RandomOverSampler()  # Replace SMOTE with RandomOverSampler for sequence data
        X, y = oversampler.fit_resample(X, y)
        print('*****************')

    initial_model = build_model(args.model_type, vocab_size, 16, maxlen, args.filters, args.units, args.dropout_rate)
    initial_weights = initial_model.get_weights()  # Save the initial weights for resetting later

    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    results = []
    RESULT = []
    all_fpr = []
    all_tpr = []
    all_thresholds = []
    all_roc_auc = []
    all_PR_auc = []

    for fold, (train_index, val_index) in enumerate(kf.split(X), 1):
        logging.debug(f"Fold {fold}...")

        try:
            # Split data into training and validation sets
            X_train_fold, X_val_fold = X[train_index], X[val_index]
            y_train_fold, y_val_fold = y[train_index], y[val_index]

            # Create tf.data.Dataset objects for training and validation
            train_dataset = create_tf_dataset(X_train_fold, y_train_fold, args.batch_size)
            val_dataset = create_tf_dataset(X_val_fold, y_val_fold, args.batch_size)

            # Reset model weights to initial weights
            #model = build_model(args.model_type, args.num_words, 16, args.maxlen, args.filters, args.units, args.dropout_rate)
            model = build_model(args.model_type, vocab_size, 16, maxlen, args.filters, args.units, args.dropout_rate)
            model.set_weights(initial_weights)

            # Define callbacks
            early_stopping = EarlyStopping(monitor='val_loss', patience=5)
            checkpoint = ModelCheckpoint(f'TEST/{args.model_type}/{args.project_name}/{args.task_name}_Apply_best_model_fold_{fold}.h5', save_best_only=True, monitor='val_loss', mode='min')
            lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.0001)

            # Training with EarlyStopping, ModelCheckpoint, and Learning Rate Scheduler
            model.fit(train_dataset, epochs=args.epochs,
                      validation_data=val_dataset,
                      callbacks=[early_stopping, checkpoint, lr_scheduler])

            val_predictions_proba = model.predict(val_dataset).ravel()
            metrics, fpr, tpr, thresholds, roc_auc, PR_auc = calculate_metrics(y_val_fold, val_predictions_proba)

            all_fpr.append(fpr)
            all_tpr.append(tpr)
            all_thresholds.append(thresholds)
            all_roc_auc.append(roc_auc)
            all_PR_auc.append(PR_auc)
            RESULT.append([args.task_name, args.target_label,fold, args.epochs, metrics['accuracy'], metrics['precision'], metrics['recall'], metrics['f1'], fpr, tpr, thresholds, roc_auc, PR_auc])

            results.append([fold, metrics['accuracy'], metrics['precision'], metrics['recall'], metrics['f1']])

        except Exception as e:
            logging.error(f"Error during training/prediction in fold {fold}: {e}")
            continue

    results = np.array(results).astype(float)
    mean_results = results.mean(axis=0)
    logging.info(f"10-fold average result: ACC={mean_results[1]}, PRE_macro={mean_results[2]}, REC_macro={mean_results[3]}, F1_macro={mean_results[4]}")

    mean_fpr, mean_tpr = average_roc_curve(all_fpr, all_tpr)
    mean_roc_auc = np.mean(all_roc_auc)
    mean_PR_auc = np.mean(all_PR_auc)

    logging.info(f"10-fold average ROC AUC: {mean_roc_auc}")
    logging.info(f"Average FPR: {mean_fpr}")
    logging.info(f"Average TPR: {mean_tpr}")
    logging.info(f"Common thresholds: {np.unique(np.concatenate(all_thresholds))}")
    RESULT.append([args.task_name, args.target_label,fold, 'Average', mean_results[1], mean_results[2], mean_results[3], mean_results[4], '', '', '', mean_roc_auc, mean_PR_auc])
    header = ['Project', 'Task', 'Fold', 'Epoch', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'fpr', 'tpr', 'thresholds', 'Roc_auc', 'PR_auc']
    write_csv(args.project_name, header, RESULT, args.stopword_flag, args.model_type, args.data_type)

    print_timestamp("Finish")

if __name__ == '__main__':
    main()