# sentiment-analysis-for-X-data
**Airline Sentiment Analysis Project**
**Overview**
This project focuses on analyzing customer sentiment from airline-related tweets. The goal is to classify tweets into three sentiment categories: negative, neutral, and positive. The analysis involves data preprocessing, sentiment classification using the DistilBERT model, and generating actionable insights for airline companies to improve customer satisfaction.

**The project is divided into two main parts:**

Data Preprocessing and Exploratory Data Analysis (EDA): This involves cleaning the dataset, handling missing values, and visualizing sentiment distribution across different airlines.

Sentiment Classification with DistilBERT: This part involves training a DistilBERT model to classify tweets based on sentiment and evaluating its performance.

**Key Features**
**1. Data Preprocessing**
Handling Missing Values: Missing values in the negativereason and negativereason_confidence columns were addressed.

Text Cleaning: Stopwords were removed, and text was tokenized and converted to lowercase.

Duplicate Removal: Duplicate tweets were identified and removed to ensure data quality.

**2. Exploratory Data Analysis (EDA)**
Sentiment Distribution: Visualized the overall distribution of sentiments (negative, neutral, positive) across the dataset.

Sentiment by Airline: Analyzed sentiment distribution for each airline to identify trends and pain points.

Word Clouds: Generated word clouds for negative and positive tweets to identify common themes and keywords.

Negative Reasons: Identified the top 10 reasons for negative sentiments, such as customer service issues, late flights, and lost luggage.

**3. Sentiment Classification with DistilBERT**
Model Training: Trained a DistilBERT model for 7 epochs to classify tweets into negative, neutral, and positive sentiments.

Model Evaluation: Achieved a test accuracy of 82.98% and analyzed precision, recall, and F1-scores for each sentiment class.

Loss and Accuracy Curves: Visualized training and validation loss/accuracy curves to monitor model performance.

**Key Findings**
Insights from EDA
Negative Sentiment Dominance: Approximately 63% of tweets express negative sentiments, with American Airlines having the highest percentage of negative tweets.

Positive Sentiment Leaders: Virgin America has the highest percentage of positive tweets, indicating better customer satisfaction.

Common Negative Reasons: The top reasons for negative tweets include customer service issues, late flights, and lost luggage.

Word Frequency: Negative tweets frequently mention words like "flight," "delay," and "service," while positive tweets include words like "thanks," "great," and "awesome."

Insights from Model Evaluation
Model Performance: The DistilBERT model achieved an accuracy of 82.98% on the test set.

**Class-wise Performance:**

Negative Sentiment: High precision (0.87) and recall (0.92).

Neutral Sentiment: Lower precision (0.70) and recall (0.60), indicating room for improvement.

Positive Sentiment: Moderate precision (0.79) and recall (0.79).

**How to Use**
Prerequisites
Python 3.10 or higher

Required Python libraries: pandas, numpy, matplotlib, seaborn, nltk, scikit-learn, torch, transformers, wordcloud

Installation
Clone the repository:

git clone https://github.com/your-username/sentiment_analysis_for_X_data.git
cd sentiment_analysis_for_X_data


**Running the Notebooks**
Data Analysis and Preprocessing:

Open the data_analysis.ipynb notebook to explore the dataset, perform EDA, and preprocess the data.

Sentiment Classification with DistilBERT:

Open the sentiment_analysis_bert.ipynb notebook to train and evaluate the DistilBERT model.

**Future Work**
Hyperparameter Tuning: Experiment with different learning rates, batch sizes, and dropout rates to improve model performance.

Advanced Models: Explore larger transformer models like BERT or RoBERTa for better sentiment classification.

Real-time Sentiment Analysis: Develop a real-time sentiment analysis system to monitor customer feedback continuously.

**License**
This project is licensed under the MIT License.

**Acknowledgments**
The dataset used in this project is sourced from Kaggle.
