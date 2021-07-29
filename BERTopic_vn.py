import utils
from bertopic import BERTopic
from flair.embeddings import TransformerDocumentEmbeddings
from sklearn.datasets import fetch_20newsgroups
docs = fetch_20newsgroups(subset='all',  remove=('headers', 'footers', 'quotes'))['data']






data = utils.read_dataset()
docs = data["text"].tolist()


roberta = TransformerDocumentEmbeddings('vinai/phobert-base')
topic_model = BERTopic(embedding_model=roberta)