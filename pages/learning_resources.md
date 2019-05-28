# Gaining background in Machine Learning and Recurrent Neural Networks

First, you need to understand the methods we are going to use to solve a cool research problem this summer. This project is under the umbrella of **Machine Learning** as we are trying to predict the labels of time series data. We are investigating a **Deep Learning** algorithm called **Recurrent Neural Networks**.

The topics you need to have a strong grasp on are:
* Basic Machine Learning
* Deep Feedforward Neural Networks
* Recurrent Neural Networks

# Reading List
* [Deep Learning Textbook](https://www.deeplearningbook.org/)
    * Part I - [Chapter 5](http://www.deeplearningbook.org/contents/ml.html)
    * Part II - [Chapter 6](http://www.deeplearningbook.org/contents/mlp.html), [Chapter 7](http://www.deeplearningbook.org/contents/regularization.html), [Chapter 10](http://www.deeplearningbook.org/contents/rnn.html)
    * More in depth:
        * Chapter 8, particularly 8.2.5
        * My interests: 10.7, 10.9, 10.11
* [The Unreasonable Effectiveness of RNNs](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)
* [Recurrent Neural Networks Tutorial](http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-1-introduction-to-rnns/)
* [Stanford cs231n](http://cs231n.github.io/): This course has very clear descriptions of complex deep learning topics. I recommend *Neural Networks* parts [1](http://cs231n.github.io/neural-networks-1/), [2](http://cs231n.github.io/neural-networks-2/), and [3](http://cs231n.github.io/neural-networks-3/).
* [Understanding LSTMs](http://colah.github.io/posts/2015-08-Understanding-LSTMs/): This is the integral update to the classic RNN that allowed for the success seen by RNNs. If you get through here and feel that you have a good enough understanding to explain the intuition of an RNN with LSTM memory cells, then you are doing fantastic!

# Videos
  * [Andrew Ng's Coursera Course](https://www.coursera.org/specializations/deep-learning): If you are unfamiliar with deep learning, please watch these videos in sequence. There is no need to pour time into the homeworks, but they can be valuable resources as well. Videos are also available on [YouTube](https://www.youtube.com/watch?v=7PiK4wtfvbA&list=PLBAGcD3siRDguyYYzhVwZ3tLvOyyG5k6K). Please watch the RNN videos, they are short.

# Technology
All of our programming will be Python. If you are unfamiliar with Python, **LEARNING PYTHON IS YOUR FIRST TASK**. Try [this tutorial](http://cs231n.github.io/python-numpy-tutorial/).
Our preprocessing will be done using [NumPy](http://www.numpy.org/) and deep learning algorithms will be implemented using [PyTorch](https://pytorch.org/).

* Programming will be in Python.
* Deep Learning algorithms will be implemented in PyTorch.
* Group work will be done in GitHub.
* Group messages will be through Slack.
* We will be using Electronic Health Records (EHR) from the [MIMIC III database](https://mimic.physionet.org/).

Here's how to get started:
* Getting started with PyTorch (assuming you have a vague idea of what deep learning is and have some experience scripting in Python).
  * I recommend beginning with the [60 Minute Blitz](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html) to get familiar with what it means to design deep learning algorithms in PyTorch.
  * Next, you need to follow along with a couple of tutorials where you implement a full deep learning pipeline (data loading, processing, defining a model, training the model, evaluating the model). For this, I recommend [Classifying Names with a Character-Level RNN](https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html), [Translation with a Sequence to Sequence Network and Attention](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html), and finally [Sequence Models and Long-Short Term Memory Networks](https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html#sphx-glr-beginner-nlp-sequence-models-tutorial-py)
  * After these tutorials, I will provide the code for my current research and we will discuss next steps.
* We will be storing our work through a GitHub repository that I will invite you to.
  * If you have not used GitHub much, please complete [this tutorial](https://guides.github.com/activities/hello-world/).
  * I will create a folder for each of us where you may store code and data.
  * You should have an understanding of the expected GitHub workflow.
* The MIMIC III database is a rather complex relational database containing clinical records of over 45,000 patients. We will be extracting a subset of the data and detecting different adverse events contained within.
  * I recommend reading through the data pages to get a hint of how the database is laid out (i.e., what information is contained? How would we extract one person's heart rate? Or uncover whether or not they died in the hospital?).
  * As it is a relational database, extracting the data is easiest via SQL, a relational query language. It is very intuitive to use, but takes a bit of exploration, please look up a SQL tutorial. Don't get started on this until we discuss, it's likely that you won't need to do this at all and I will just send you a dataset.

Some useful specific stuff:
* Copying files to/from a remote server (e.g. Turing)? Use [Rsync](https://www.tecmint.com/rsync-local-remote-file-synchronization-commands/).
* Searching through your history of terminal commands to repeat a command? Use [Fuzzy Finder](https://github.com/junegunn/fzf).
