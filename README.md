# Clinical Outcome Prediction using Recurrent Neural Networks

WPI Data Science REU, May to August 2019. *Page author: Tom Hartvigsen*.

This summer we will be implementing Recurrent Neural Networks (RNN) to make clinical predictions using multivariate time series.
In general, we will begin with a lot of reading and learning, then move to a crisp definition of our problem with a sequence of tasks and checkpoints, then finally implement the model and evaluate it on a few datasets and compare against state-of-the-art methods, leading to a publication.

Our goal is to write a paper based on the work we do during **only these 10 weeks**, so speed will be essential.

The first couple of weeks will be a learning phase during which you will build an understanding of general Machine Learning, Deep Learning, and specifically Recurrent Neural Networks.
You will likely come across terms that you (and maybe I) don't understand, so it will be important to efficiently select which rabbit-holes we go down.
I have been working on Machine Learning research for the past three years and specifically RNNs for the past two years, so I will do my best to be a resource for you.

## Project Description

Our goal for the summer is to develop a state-of-the-art Machine Learning model that predicts which infections patients will likely develop during their stays in a hospital.
These data are messy (many missing values leading to sparse data) and infection-prediction is incredibly complex (even doctors struggle with some of these tasks!).
This makes classification challenging and we often resort to vast simplifications of these data and the problem in order to actually solve something.
In cases such as these, it seems that there can be [better ways](/papers) to handle these data.
To accomplish this, we will be modifying the architecture of RNNs, a fascinating machine learning model.

RNNs are the state-of-the-art method for many sequential learning tasks.
However, training them is precarious and is made more challenging in the clinical domain where missing values are common.
We will attempt to modify the RNN architecture such that some challenges associated with specifically-clinical data are less of a stumbling block, thus bringing the success found by RNNs in "cleaner" domains into the clinical domain.

The data we will use is from the [MIMIC III Intensive Care Unit Database](https://mimic.physionet.org/) which contains Electronic Health Records (EHR) for over 45,000 patients over 11 years in Beth Israel Deaconess Medical Center. We will extract sequences of tests and medications and recordings for a subset of these 45,000 patients. The extraction will be done using SQL, an intuitive programming language for extracting data from relational databases.

## Links
* [Tutorials](/pages/tutorials/turing_jobs.md)
* [Software/tools we will be using](/pages/tools.md)
* [Learning Resources](/pages/learning_resources.md)
* [Papers](/papers)
* [FAQs](/pages/faq.md)

## First Steps
* Take a first glance over all links I have provided in this repository and send initial questions my way.
* Request access to the [MIMIC III database](https://mimic.physionet.org/gettingstarted/access/), list Elke Rundensteiner as your advisor.
* Go through the Learning Resources links to become familiar with the algorithms with which we will work.
* Read the first paper I linked in [papers](/papers) (we will be implementing their algorithm first) and summarize the **high level** ideas of the paper (for example, what is the problem they are trying to solve? What data do they use? What is the output of the model?).

## Expected Learning Outcomes
The goal is to discover the best solution to a well-formed problem, thus having publishable results. Along the way, you will become proficient in the following skills:
* Implementing Deep Learning algorithms in PyTorch.
* Deep understanding of Recurrent Neural Networks and related topics.
* Understanding the flow of research (hopefully you will better understand whether or not you are interested in conducting more research in the future!).
* Effective research communication.
* Data extraction from relational databases via SQL (possibly).

## Who is Tom?

I am beginning my fourth year in the Data Science Ph.D. program at WPI, working with Dr. Elke Rundensteiner and Xiangnan Kong. You can find more information on [my website](https://thartvigsen.github.io/), but in general I am interested in a few specific topics: Representation Learning for Sequences (Recurrent Neural Networks), World Models, and Conditional Computing. I have focused on Machine Learning in the clinical domain for my first three years at WPI, but am not particularly tied to any domain. Instead, my main interests are mostly theoretical by nature.

Please use the following email addresses to reach me:
* twhartvigsen@wpi.edu (primary email)
* twhartvigsen@gmail.com (for google-related links (e.g., google hangouts))
