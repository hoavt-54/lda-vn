This is another topic extraction demo using LDA and [BERTopic](https://github.com/MaartenGr/BERTopic) but for the Vietnamese language. There are many dataset available online, feel free to pick one on your own. Here's a [sample dataset containing 28k documents](https://dataset.duyet.net/vnexpress/vnexpress-28k-posts.zip).
## How to run this code
After clone the project to your computer and download a dataset, you need to extract and put the dataset in to [data/data_vn](data/data_vn):

<pre>
mkdir data/
unzip ~/path/to/the/dataset/vnexpress-28k-posts.zip -d data
mv data/phan1 data/data_vn
rm vnexpress-28k-posts.zip
</pre>

Install prerequisite

<pre>
virtualenv env -p python3
pip install -r requirements.txt
git submodule init & git submodule update
</pre>

Run LDA and its visualization
<pre>
python3 lda_vn.py
open lda_vizs/lda_visualization_40.html
</pre>

Run BERTopic and its visualization (This takes a while on the first time running.)

<pre>
python3 BERTopic_vn.py
open saved_models/bertopic_vnmese_plot_20.html
open saved_models/bertopic_vnmese_barchat_20.html
</pre>