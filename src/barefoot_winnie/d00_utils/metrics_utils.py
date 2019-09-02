from nltk.translate.bleu_score import SmoothingFunction
import nltk
import nltk.translate.bleu_score as bleu
from multiprocessing import cpu_count
from gensim.corpora import Dictionary
from gensim.models import Word2Vec
from gensim.similarities import MatrixSimilarity
from gensim.matutils import Dense2Corpus
from gensim.matutils import softcossim
from scipy import sparse
from textdistance import jaro_winkler,jaccard,monge_elkan,overlap


def produce_similarity_score_text(text_1, text_2, metric='edit_distance'):
    """similarity between two strings"""
    if metric == 'edit_distance':
        similarity_score = nltk.edit_distance(text_1, text_2)
    if metric == 'bleu':
        tokenized_1 = text_1.split()
        tokenized_2 = text_2.split()
        short_sentence_smoother = SmoothingFunction().method4
        similarity_score = bleu.sentence_bleu([tokenized_1], tokenized_2, smoothing_function=short_sentence_smoother)
    if metric == 'jaro':
        similarity_score = jaro_winkler.normalized_similarity(text_1,text_2)
    if metric == 'jaccard':
        similarity_score = jaccard.normalized_similarity(text_1, text_2)
    if metric == 'monge_elkan':
        similarity_score = monge_elkan.normalized_similarity(text_1, text_2)
    if metric == 'overlap':
        similarity_score = overlap.normalized_similarity(text_1, text_2)
    return similarity_score


def soft_cosine_similarity(text_1, text_2, corpus):
    dictionary = Dictionary(corpus)
    text_1 = dictionary.doc2bow(text_1)
    text_2 = dictionary.doc2bow(text_2)
    w2v_model = Word2Vec(corpus, workers=cpu_count(), min_count=1, size=300, seed=12345)
    similarity_matrix = sparse.csr_matrix(MatrixSimilarity(Dense2Corpus(w2v_model.wv.syn0.T)))
    return softcossim(text_1, text_2, similarity_matrix)
