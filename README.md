# we3tasks

we3tasks is a simple way to evaluate Metonymi's representations for yourself. We provide Metonymi's representations
for 3 different document corpuses and we3tasks builds a simple classifier on top of the representations
for each document corpus.

# The Tasks

The datasets we'll examine are 20newsgroups (document classification), Reuters (document classification), and the
Large Movie Review Dataset (sentiment analysis). Metonymi's technology has never been trained on or examined any of these datasets, but the representations it creates for the documents have predictive power for each of the tasks.

# Setup

We recommend using a virtual environment. Inside the we3tasks repo run

``` bash
virtualenv metonymi
source metonymi/bin/activate
```

 and then install the requirements:

 ```bash
 pip install -r requirements.txt
 ```

 If you don't want to use virtualenv, just run the pip command.

 # Data Assembly

 To see how the datasets are employed, run

 ``` bash
 python scripts/20newsgroup_assembly.py
 python scripts/reuters_assembly.py
 python scripts/aclImdb_assembly.py
 ```

 which produce separate folders in we3tasks for each of the tasks. These scripts download the datasets and then
 lightly clean them. The Reuters dataset has documents that can belong in overlapping categories. For the classification
 task, we only use those documents assigned to a single category.

 # Metonymi features

Download the Metonymi representations for each dataset from the following links

[20newsgroups](https://github.com/imodpasteur/lutorpy/issues/38)
[Reuters](https://github.com/imodpasteur/lutorpy/issues/38)
[aclImdb](https://github.com/imodpasteur/lutorpy/issues/38)\

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
to perform the classification tasks using the Metonymi features.

# Discussion

As you can see when you run benchmarl.py, the classification rates for these tasks are as follows:


Remember, these are fairly simple tasks: tf-idf or bag of words features probably outperform our features for these tasks.
However, elementary methods such as tf-idf are going to fail on more complex semantic tasks and our features can always supplement
them. The point is that Metonymi's representations work on a variety of tasks at once.
