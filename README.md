# Classifying Multivariate Time Series with Missing Values

WPI Data Science REU, May to August 2018. *Page author: Tom Hartvigsen*.

This summer we will be implementing Recurrent Neural Networks (RNN) to solve multivariate time series problems.
In general, we will begin with a lot of reading and learning, then move to a crisp definition of our problem with a sequence of tasks and checkpoints, then finally implement the model and evaluate it on a few datasets and compare against state-of-the-art methods, leading to a publication.

Our goal is to write a paper based on the work we do during **only these 10 weeks**, so speed will be essential.

The first couple of weeks will be a learning phase during which you will build an understanding of general Machine Learning, Deep Learning, and specifically Recurrent Neural Networks.
You will likely come across terms that you (and maybe I) don't understand, so it will be important to efficiently select which rabbit-holes we go down.
I have been working on Machine Learning research for the past two years and specifically RNNs for the past year, so I will do my best to be a resource for you.

## Project Description

Our goal for the summer is to develop a state-of-the-art Machine Learning model to classify Multivariate Time Series with Missing Values.
Some domains are more prone to missing values than others (e.g., clinicians not-requesting a particular set of measurements leads to a large number of missing values).
This makes classification challenging and we often resort to vast simplifications of these data.
For instance, mean-value imputation.
In cases such as these, it seems that there can be [better ways](/papers) to handle these data.
To accomplish this, we will be modifying the architecture of RNNs, a fascinating Deep Learning model.

RNNs are the state-of-the-art method for many tasks involving real-valued sequential data.
However, training them is precarious and is made more challenging when missing values are frequent.
We will attempt to modify the RNN architecture such that missing values are less of a stumbling block, thus bringing the success of RNNs into new domains, a promising direction in the age of big data.

The data we will use is from the [MIMIC III Intensive Care Unit Database](https://mimic.physionet.org/) which contains Electronic Health Records (EHR) for over 45,000 patients over 11 years in Beth Israel Deaconess Medical Center. We will extract sequences of tests and medications and recordings for a subset of these 45,000 patients. The extraction will be done using SQL, an intuitive programming language for extracting data from relational databases.

## Links
* [Software/tools we will be using](/pages/tools.md)
* [Learning Resources](/pages/learning_resources.md)
* [Papers](/papers)
* [FAQs](/pages/faq.md)

## First Steps
* Read through the links I have provided and send any questions you have my way.
* Request access to the [MIMIC III database](https://mimic.physionet.org/gettingstarted/access/).
* Go through the Learning Resources links to become familiar with the algorithms with which we will work.
* Read the paper I linked in [papers](/papers) (we will be implementing their algorithm first) and summarize their work in a slideshow.

## Summer Learning Outcomes
The goal is to discover the best solution to a well-formed problem, and thus having publishable results. Along the way, you will become proficient in the following skills:
* Deep Learning algorithm implementations in PyTorch.
* Deep understanding of Recurrent Neural Networks and related topics.
* Understanding the flow of research (hopefully you will better understand whether or not you are interested in conducting more research in the future!).
* Data extraction from relational databases via SQL (possibly).

## Who is Tom?

I am beginning my third year in the Data Science Ph.D. program at WPI, working with Dr. Elke Rundensteiner. You can find more information on [my website](https://thartvigsen.github.io/), but in general I am interested in a few topics: Recurrent Neural Networks, Conditional Computing, and complex long-term dependencies in sequential data. I have focused on Machine Learning in the clinical domain for my first two years at WPI, but am not particularly interested in any domain. Instead, my main interests lie in recurrent neural network theory.

Please contact me at the following locations:
* twhartvigsen@wpi.edu (primary email)
* twhartvigsen@gmail.com (for google-related links (e.g., google hangouts))
