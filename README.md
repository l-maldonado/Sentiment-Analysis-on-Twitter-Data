# Sentiment Analysis on Twitter Data

This project explores various methods of sentiment analysis on a dataset of Twitter posts. The objective is to compare the performance of different sentiment analysis techniques, such as lexicon-based approaches, machine learning models, and deep learning techniques. The analysis includes evaluating accuracy, precision, recall, and F1-score for each method.

A full explanation to the project is available here: [https://l-maldonado.github.io/Sentiment-Analysis-on-Twitter-Data](https://l-maldonado.github.io/Sentiment-Analysis-on-Twitter-Data/)

## Table of Contents

1. [Introduction](#introduction)
2. [Installation and Setup](#installation-and-setup)
3. [Usage](#usage)
4. [License](#license)

---

## Introduction

Sentiment analysis is a crucial technique in natural language processing (NLP) used to determine the sentiment or emotional tone of a piece of text. This project focuses on analyzing Twitter data to classify tweets as positive, negative, or neutral. Several sentiment analysis techniques are implemented and compared to identify the most effective approach for this specific dataset.

This project is structured into three modules: Advanced Data Analytics, Presentation and Visualization, and Big Data Storage and Processing. The aim is to analyze Twitter data, perform sentiment analysis, visualize trends, and build predictive models.


### Dataset

The dataset used for this project consists of tweets produced over a fixed period of time and gathered using the Twitter API. It includes information such as tweet text, user details, and timestamp. The data is preprocessed to remove irrelevant content like URLs, mentions, and hashtags, and is then used for sentiment classification tasks.


### Project Structure

    /sentiment-analysis-project
    ├── /data
    │ ├── ProjectTweets.csv         # Raw Twitter dataset
    │ ├── ProjectTweets (1).csv     # Cleaned dataset
    │ ├── ProjectTweets (2).csv     # NLP labeled tokenized indexed dataset
    │ ├── positive-words.txt        # English labeling words dataset
    │ ├── negative-words.txt        # English labeling words dataset
    ├── /images
    │ ├── Twitter Project - Structure.drawio.png
    │ ├── Presentation poster.jpg
    │ ├── Misc plots
    ├── /notebooks                  # Jupyter notebook for analysis and visualizations
    │ ├── Twitter Data Mining and Sentiment Analysis - Part 1 Cleaning and EDA.ipynb
    │ ├── Twitter Data Mining and Sentiment Analysis - Part 2 NLP Processing and Visualization.ipynb
    │ ├── Twitter Data Mining and Sentiment Analysis - Part 3 Apply Assess and Compare Models.ipynb
    │ ├── Twitter Data Mining and Sentiment Analysis - Part 4 Big Data Analytics Notebook.ipynb
    ├── /src
    │ ├── preprocess.py             # Data preprocessing scripts
    │ ├── lexicon_analysis.py       # Lexicon-based sentiment analysis
    │ ├── ml_model.py               # Machine learning sentiment analysis model
    │ ├── dl_model.py               # Deep learning sentiment analysis model
    ├── FINAL REPORT_Twitter Data Mining and Sentiment Analysis.pdf
    ├── README.md                   # Project documentation
    ├── requirements.txt            # Python dependencies
    ├── LICENSE.md                  # Project licencing


## Installation and Setup

### Prerequisites

Make sure you have Python 3.7+ installed. You will also need to install the following libraries:

- pandas
- numpy
- scikit-learn
- nltk
- tensorflow
- keras
- matplotlib
- seaborn
- vaderSentiment
- textblob

### Installation Steps

1.  Clone this repository:

    ```bash
    git clone https://github.com/username/sentiment-analysis-twitter.git
    cd sentiment-analysis-twitter

    ```

2.  Install the required dependencies:

        pip install -r requirements.txt

3.  Download the Twitter dataset (if applicable):

    Follow the instructions in the /data folder to acquire the dataset.

## Usage

**Running the Jupyter Notebook**

To explore and run the analysis, open the Jupyter notebook sentiment_analysis.ipynb located in the /notebooks folder.

        jupyter notebook

You can run each cell in the notebook to:

1. Load and preprocess the data
2. Perform sentiment analysis using different methods
3. Visualize the results

**Running the Scripts**

To execute the individual sentiment analysis scripts, you can run the Python files in the /src folder directly:

        python src/lexicon_analysis.py
        python src/ml_model.py
        python src/dl_model.py

## License

All rights reserved

No Use Without Permission

No Commercial Use

Modification Prohibited

Attribution must be made

See the [LICENSE](LICENSE.md) file for details.

---
