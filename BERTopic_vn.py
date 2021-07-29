from vncorenlp import VnCoreNLP
from BERTopic.bertopic import BERTopic
from flair.embeddings import TransformerDocumentEmbeddings
from utils import read_dataset, Tokenizer, cache_tokenized, read_tokenized


SAVED_MODEL = "saved_models/vietnamese_vnexpress_%s.model"
SAVED_PLOT = "saved_models/bertopic_vnmese_plot_%s.html"
SAVED_BAR = "saved_models/bertopic_vnmese_barchat_%s.html"
EMBS = "vinai/phobert-base"

def build_model(tokenizer, min_topic_size=15):
    docs = read_tokenized()
    if not docs:
        data = read_dataset()
        docs = data["text"].tolist()
        docs = [" ".join(tokenizer.preprocess(doc, filter=False)) for doc in docs]
        cache_tokenized(docs)
    
    phobert = TransformerDocumentEmbeddings(EMBS)
    topic_model = BERTopic(
        embedding_model=phobert,
        min_topic_size=min_topic_size
        )
    topics, _ = topic_model.fit_transform(docs)

    topic_model.get_topic_info()
    topic_model.save(SAVED_MODEL % min_topic_size)
    topic_model.visualize_topics().write_html(SAVED_PLOT % min_topic_size)
    topic_model.visualize_barchart().write_html(SAVED_BAR % min_topic_size)


def inference():
    topic_model = BERTopic.load(SAVED_MODEL)


def main():
    annotator = VnCoreNLP('VnCoreNLP/VnCoreNLP-1.1.1.jar', annotators="wseg,pos", max_heap_size='-Xmx3g')
    tokenizer = Tokenizer(annotator)
    build_model(tokenizer, min_topic_size=20)


if __name__ == "__main__":
    main()

