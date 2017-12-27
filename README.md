# we3tasks

we3tasks is a simple way to evaluate Metonymi's representations for yourself. We provide Metonymi's representations
for 3 different document corpuses and we3tasks builds a simple classifier on top of the representations
for each document corpus.

# The tasks

The datasets we'll examine are 20newsgroups (document classification), Reuters (document classification), and the
[Large Movie Review Dataset](http://ai.stanford.edu/~amaas/data/sentiment/) (sentiment analysis). Metonymi's technology has never been trained on or examined any of these datasets, but the representations it creates for the documents have predictive power for each of the tasks.

# Setup

We recommend using a virtual environment. Clone the repo and create a virtual environment inside it:

``` bash
git clone https://github.com/Metonymi/we3tasks.git
virtualenv metonymi

Activate environment with

``` bash
source metonymi/bin/activate

or Windows:

``` bash
Scripts\activate


 and then install the requirements:

 ```bash
 pip install -r requirements.txt
 ```

 If you don't want to use virtualenv, just run the pip command.

 # Data assembly

 To see how the datasets are employed, examine and run

 ``` bash
 python main_assembly.py
 ```

 which produce separate folders in we3tasks for each of the tasks. These scripts download the datasets and then
 lightly clean them. The Reuters dataset has documents that can belong in overlapping categories, so for the Reuters
 task, we only use those documents assigned to a single category. This has the interesting side effect of creating a highly
 imbalanced dataset for testing.

 # Metonymi features

Download and extract the Metonymi representations for the datasets [here](https://s3-us-west-2.amazonaws.com/metonymipublic/we3tasks_features12.tar.gz) inside the we3tasks repo.

These files are torch files and can be inspected in python as follows:

```python
import torchfile

features = torchfile.load('/path/to/20newsgroup_data_features')
```

Alternatively, if you have Metonymi credentials and don't want to use torchfile,
you can just zip the folders produced by the assembly scripts and pass them to our API with cURL. For example,

```bash
curl --user your_user_name:your_password -F "file=@/path/to/we3tasks/20newsgroups.zip" -F "title=20newsgroup" https://api.metonymi.ai/process/uploads/upload_file
```

These files won't be nicely formatted for you. The Metonymi API doesn't care about the path structure of the uploaded folders.

# Linear classifier

With the representation files in we3tasks, just run the benchmark script:

```bash
python benchmark.py
```

This script assembles the representations and the raw text data into a single object, and then trains a GLM with sklearn
to perform the classification tasks using the Metonymi features. Some vital statistics for a test set are also returned.

# Discussion

As you can see, Metonymi's representations have dimension 2048, which is small for NLP applications. We recommended adding
your own features to Metonymi's for the best possible model (make sure to increase the training iterations of whatever
algorithm you chose to accommodate the increased dimensionality if you use your own features).

We think the best way to use this repo is as a proof of concept: these tasks can be performed successfully using classic TF-IDF
features, but those features are going to let you down if the task is [more complicated](https://github.com/niderhoff/nlp-datasets) than these are. That's where Metonymi's features are going to help you the most. We're not looking at simple statistics about word use; we're creating a distributed representation for each document that captures its complex semantic properties and puts individual words in a greater context.
