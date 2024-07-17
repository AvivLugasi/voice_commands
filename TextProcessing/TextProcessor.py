import string
import nltk
from gensim.models.word2vec import Text8Corpus
from gensim.test.utils import datapath
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from gensim.models.phrases import Phrases, Phraser, ENGLISH_CONNECTOR_WORDS

nltk.download('punkt')
nltk.download('stopwords')


class TextProcessor:
    def __init__(self,
                 tokenize=True,
                 case_folding=True,
                 remove_punctuation=True,
                 remove_stopwords=True,
                 use_lemmatizer=True):
        self.sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.tokenize = tokenize
        self.case_folding = case_folding
        self.remove_punctuation = remove_punctuation
        self.remove_stopwords = remove_stopwords
        self.use_lemmatizer = use_lemmatizer

    def process_text(self, written_speach: str):
        processed_text = []
        if self.tokenize:
            text = self.tokenize_text(written_speach)
        for sentence in text:
            processed_sentence = []
            for token in sentence:
                if self.remove_stopwords and token in self.stop_words:
                    continue
                if self.case_folding:
                    token = _case_folding(token)
                if self.remove_punctuation:
                    token = _remove_punctuation(token)
                processed_sentence.append(token)
            # # Train the bigram model
            # phrase_model = Phrases(text,
            #                        min_count=1,
            #                        threshold=10,
            #                        delimiter='_',
            #                        connector_words=ENGLISH_CONNECTOR_WORDS)
            #
            # # Transform sentences to include bigrams
            # bigram_sentences = phrase_model[processed_sentence]
            # #print(bigram_sentences)
            # processed_sentence = bigram_sentences
            if self.use_lemmatizer:
                pos_tags = pos_tag(processed_sentence)
                processed_sentence = [self.lemmatize(word, pos=_get_wordnet_pos(tag)) for word, tag in pos_tags]
            processed_text.append(' '.join(processed_sentence))
        return processed_text

    def tokenize_text(self, written_speach: str):
        splitted_sentences = self.tokenize_to_sentences(written_speach)
        return _tokenize_to_words(splitted_sentences)

    def tokenize_to_sentences(self, written_speach: str):
        return self.sent_detector.tokenize(written_speach.strip(), realign_boundaries=False)

    def lemmatize(self, word: str, pos: str):
        if word is not None and pos is not None:
            return self.lemmatizer.lemmatize(word, pos=pos)
        return word


def _get_wordnet_pos(tag):
    if tag.startswith('J'):
        return 'a'  # Adjective
    elif tag.startswith('V'):
        return 'v'  # Verb
    elif tag.startswith('N'):
        return 'n'  # Noun
    elif tag.startswith('R'):
        return 'r'  # Adverb
    else:
        return None  # default to noun


def _case_folding(token: str):
    """
    Convert a token to lower case and return it
    """
    # lower case
    return token.lower()


def _tokenize_to_words(splitted_sentences: str):
    tokenized_text = []
    for sentence in splitted_sentences:
        tokenized_text.append(sentence.split())
    return tokenized_text


def _remove_punctuation(token: str):
    return token.translate(str.maketrans('', '', string.punctuation))

#
# processor = TextProcessor()
text = '''
Gold open and clear. Open and clear use shotgun.
Arrest suspect. Zip the suspect.
Stack up.
Breach and clear use c2.
Breaching using shotgun.
Police dont move.
hand's up LSPD.
LSPD swat, dont move!
tie up.
gold team open and clear.
open and clear use explosive.
open use shotgun.
new york police, dont move.
trees graph and linked lists.
good morning.
he went to bank of america.
use CS gas and clear.
use tear gas and clear
'''
# processed_sentences = processor.process_text(text)
#
# print(processed_sentences)
