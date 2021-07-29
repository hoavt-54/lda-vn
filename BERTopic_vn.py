from vncorenlp import VnCoreNLP
from bertopic import BERTopic
from flair.embeddings import TransformerDocumentEmbeddings
from utils import read_dataset, Tokenizer


annotator = VnCoreNLP('VnCoreNLP/VnCoreNLP-1.1.1.jar', annotators="wseg,pos", max_heap_size='-Xmx2g')
tk = Tokenizer(annotator)

data = read_dataset()
docs = data["text"].tolist()
docs = [" ".join(tk.preprocess(doc, filter=False)) for doc in docs]

phobert = TransformerDocumentEmbeddings('vinai/phobert-base')
topic_model = BERTopic(embedding_model=phobert)
topics, _ = topic_model.fit_transform(docs)

topic_model.get_topic_info()
topic_model.save("saved_models/vietnamese_vnexpress_2.model")


## inference