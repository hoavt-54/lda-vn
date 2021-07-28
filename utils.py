import traceback
from matplotlib import pyplot as plt
from pathlib import Path
import pandas as pd

def read_dataset(data_dir="data/data_vn"):
    data = []
    dirpath = Path(data_dir)
    for text_file in dirpath.iterdir():
        data.append(text_file.read_text())

    return pd.DataFrame(data, columns=["text"])


def file2list(file_name):
    l = []
    with open(file_name) as f:
        for line in f:
            l.append(line.strip())
    return l


def read_text_file(fn="test.txt"):
    with open(f"data/{fn}") as f:
        text = f.read()
        return pd.DataFrame([text], columns=["text"])


class Tokenizer():

    def __init__(self, annotator) -> None:
        self.annotator = annotator
        self.stoppos = {w.strip() : 1 for w in file2list("saved/stoppos.txt")}
        self.stopwords = {w.strip() : 1 for w in file2list("saved/stopwords.txt")}


    def preprocess(self, text, filter=True):
        out = []
        try:
            annotated = self.annotator.annotate(text)
            for sentence in annotated["sentences"]:
                #print("sentence: ", sentence)
                formated = [w["form"].lower() for w in sentence if not filter or w["posTag"] not in self.stoppos]
                formated = [w for w in formated if not filter or w not in self.stopwords]
                out.extend(formated)
        except:
            traceback.print_exc()
        return out



def plot_document_dist(lda_model, corpus, num_topics=20):
    """ Show a distribution of topics in a document
        each topic name is replaced by its top words

    """
    topic_vector = lda_model[corpus[0]]     
    topic_vector = sorted(topic_vector, key = lambda x: x[1], reverse=True)
    topics = lda_model.show_topics(num_topics=num_topics, formatted=False)
    def top_words(topic):
        return "\n".join(map(lambda x: x[0], topic[:4]))
    topic_top_words = {topic[0] : f"Topic-{topic[0]}\n" + top_words(topic[1]) + "\n..." for topic in topics}

    # plot
    topic_vector = topic_vector[:5]
    x_values = [topic_top_words[t[0]] for t in topic_vector]
    x_idx = range(len(x_values))
    y_values = [t[1] for t in topic_vector]
    plt.bar(x_idx, y_values)
    plt.xticks(x_idx, x_values)
    plt.title("Distribution of topic in a document")
    plt.ylabel("Probability")
    plt.tight_layout()
    plt.xlabel("Topics")
    plt.show()

