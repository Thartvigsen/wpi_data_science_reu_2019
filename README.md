# Clinical Outcome Prediction using Recurrent Neural Networks

WPI Data Science REU, May to August 2019. *Page author: Tom Hartvigsen -- twhartvigsen@wpi.edu*.

This summer we will be implementing Recurrent Neural Networks (RNN) to predict clinical outcomes given time series collected from patients in a hospital.
First, we will read and learn about the tools we need to solve this problem.
Second, we will construct a crisp definition of our problem with a sequence of tasks and checkpoints.
Third, we will finally implement the model and evaluate it on a few datasets and compare against state-of-the-art methods, leading to a publication.

Our goal is to write a paper based on the work we do during **only these 10 weeks**, so speed will be essential.

During the learning phase, you will build an understanding of general Machine Learning, Deep Learning, and specifically Recurrent Neural Networks.
You will likely come across terms that you (and maybe I) don't understand, so it will be important to efficiently select which rabbit-holes we go down.

## Project Description

Our goal for the summer is to develop a state-of-the-art Machine Learning model that predicts which infections patients will likely develop during their stays in a hospital.
These data are messy (many missing values and lots of variables) and infection-prediction is incredibly complex (doctors struggle with this task!).
This makes classification challenging and we often resort to vast simplifications of these data and the problem in order to actually solve something.

To accomplish our goal, we will be developing a new clinical classifier based on RNNs, a fascinating family of deep learning models designed for sequences.
However, training RNNs can be precarious and is made more challenging in the clinical domain where missing values are common.
We will address some challenges associated with specifically-clinical data to lower the hurdles of applying these powerful methods to the clinical domain.

The data we will use is from the [MIMIC III Intensive Care Unit Database](https://mimic.physionet.org/) which contains Electronic Health Records (EHR) for over 45,000 patients over 11 years in Beth Israel Deaconess Medical Center. We will extract sequences of tests and medications and recordings for a subset of these 45,000 patients. The extraction will be done using SQL, an intuitive programming language for extracting data from relational databases.

## Links
* [Tutorials](/pages/tutorials/turing_jobs.md)
* [Software/tools we will be using](/pages/tools.md)
* [Learning Resources](/pages/learning_resources.md)
* [Papers](/papers)
* [FAQs](/pages/faq.md)

## First Steps
* Take a first glance over all links I have provided in this repository and send initial questions my way.
* Request access to the [MIMIC III database](https://mimic.physionet.org/gettingstarted/access/), listing Elke Rundensteiner as your advisor.
* Go through the Learning Resources links to begin familiarizing yourself with the models.
* Read through the first paper I linked in [papers](/papers) (we will be implementing their method first) and summarize the **high level** ideas of the paper (for example, what is the problem they are trying to solve? What data do they use? What is the output of the model?).

## Summer Learning Objectives
The goal is to discover the best solution to a well-formed problem, thus having publishable results. Along the way, you will become proficient in the following skills:
* Implementing Deep Learning algorithms in PyTorch.
* Deep understanding of Recurrent Neural Networks and related topics.
* Understanding the flow of research (hopefully you will better understand whether or not you are interested in conducting more research in the future!).
* Effective research communication.
* Data extraction from relational databases via SQL (possibly).

## Who is Tom?

I am beginning my sixth year in the Data Science Ph.D. program at WPI, working with Dr. Elke Rundensteiner and Xiangnan Kong. You can find more information on [my website](https://thartvigsen.github.io/), but in general I am interested in developing machine learning and data mining methods for time series and text.

Please use the following email addresses to reach me:
* twhartvigsen@wpi.edu (primary email)
* twhartvigsen@gmail.com (for google-related links (e.g., google hangouts))
