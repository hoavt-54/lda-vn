"""
Performs two things. First, visualize the dataset in terms of lengths, words
Second, running lda over this dataset which involve tokenization, remove stopwords
"""
import logging
import pathlib
from utils import Tokenizer, read_dataset
import plotly.express as px
from vncorenlp import VnCoreNLP

from gensim import models, corpora
from gensim.models import Phrases, LdaModel
from gensim.test.utils import datapath
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
import utils


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger =logging.getLogger(__name__)


VOCAB_SIZE=10000
SAVING_DIR = "saved_models"
UNIGRAM_FILE = "saved_models/unigram.data"


def preprocess(df, annotator):
    df["doc_len"] = df["text"].astype(str).apply(len)
    #px.histogram(df, x=df["doc_len"]).show()
    df = df[df["doc_len"] < 10000]
    df = df[df["doc_len"] > 300]
    #df = df.head(300)

    # tokenization, part-of-speech tagging, remove stopwords
    tk = Tokenizer(annotator)
    df["tokens"] = df["text"].astype(str).apply(tk.preprocess)
    return df["tokens"].tolist()


def build_model(annotator, num_topics=30):
    """ build and save models
    """
    df = read_dataset()

    # preprocessing: remove too frequent words, stopwords ...
    logger.info("Start preprocessing, this will take quite some time ...")
    list_list_tokens = preprocess(df, annotator)

    id2word = corpora.Dictionary(list_list_tokens)
    id2word.filter_extremes(no_below=50, no_above=0.6, keep_n=VOCAB_SIZE)
    logger.info(f"Done processing dataset len, vocab len {len(id2word.keys())}, {len(list_list_tokens)}")
    
    # convert data into df vectors
    corpus = [id2word.doc2bow(tokens) for tokens in list_list_tokens]

    for num_topics in range(10, 200, 6):
        lda_model = LdaModel(corpus, num_topics=num_topics,
                                id2word=id2word,
                                passes=2,  # 20
                                iterations=40,  # 400
                                # alpha=[0.01]*num_topics,
                                alpha="auto",
                                # eta=[0.01] * VOCAB_SIZE,
                                eta="auto")
        
        # save the model
        path = pathlib.Path(f"{SAVING_DIR}/lda_topic_{num_topics}")
        path.mkdir(parents=True, exist_ok=True)
        path = path / "lda.model"
        lda_model.save(str(path.absolute()))
        id2word.save(UNIGRAM_FILE)

        # visualize topics by LDAviz
        vis = gensimvis.prepare(topic_model=lda_model, corpus=corpus, dictionary=id2word)
        pathlib.Path("lda_vizs").mkdir(parents=True, exist_ok=True)
        pyLDAvis.save_html(vis, f'lda_vizs/lda_visualization_{num_topics}.html')
    return id2word, lda_model


def inference(
            annotator,
            filename="document1.txt",
            id2word=None,
            lda_model=None,
            num_topics=30,
            ):
    """ infer topic dist of a document given a previously trained model

    """
    
    if not id2word:
        id2word = corpora.Dictionary.load(UNIGRAM_FILE)
    
    if not lda_model:
        path = pathlib.Path(f"{SAVING_DIR}/lda_topic_{num_topics}") #  there are also other models
        path = path / "lda.model"
        lda_model = LdaModel.load(str(path))


    data = utils.read_text_file(filename)
    list_of_tokens = preprocess(data, annotator)
    text2bow = [id2word.doc2bow(text) for text in list_of_tokens]

    utils.plot_document_dist(lda_model, text2bow, num_topics)


def main():
    annotator = VnCoreNLP('VnCoreNLP/VnCoreNLP-1.1.1.jar', annotators="wseg,pos", max_heap_size='-Xmx2g')
    build_model(annotator)
    inference(annotator, num_topics=40)


if __name__ == "__main__":
    main()