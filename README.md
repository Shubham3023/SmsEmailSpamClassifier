# SMS EMAIL Spam Classification
<div>
<img src="https://github.com/Shubham3023/SmsEmailSpamClassifier/blob/main/Notebook/image.jpg" width="700" height="400" alt="SMS Spam Classification"/>
</div>


### Problem Statement
Features are computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. They describe characteristics of the cell nuclei present in the image.

This is a Binary Classification problem, in which the affirmative class (one) indicates that the message is spam, while the negative class(zero)
indicates that the message is ham i.e. Normal.

### Attributes
The collection is composed by just one text file, where each line has the correct class followed by the raw message. Some examples are listed below:
1. ham What you doing?how are you?
2. ham Ok lar... Joking wif u oni...
3. ham dun say so early hor... U c already then say...
4. spam FreeMsg: Txt: CALL to No: 86888 & claim your reward of 3 hours talk time to use from your phone now! ubscribe6GBP/ mnth inc 3hrs 16 stop?txtStop
5. spam Sunshine Quiz! Win a super Sony DVD recorder if you canname the capital of Australia? Text MQUIZ to 82277. B
6. spam URGENT! Your Mobile No 07808726822 was awarded a L2,000 Bonus Caller Prize on 02/09/03! This is our 2nd attempt to contact YOU! Call 0871-872-9758 BOX95QU

### Solution Proposed 
- Here model is trained on SMS spam dataset
- Same code can be used for Email classification with same model or with model trained on Email spam dataset.
In this project, the affirmative class (one) indicates that the message is spam, while the negative class(zero)
indicates that the message is ham i.e. Normal.

The aim is to correctly classify the messages and reduce False Positive.
False Positive => The message is ham but model prediction is spam. This will lead to receipt of important messages.

The following steps are performed in Jupyter notebook:
Notebook link:

1. Basic data Cleaning and EDA
2. Data pre-processing (lowercase, tokenisation, special character removal, stemming)
3. Model Building

3.1 Label Encoding
3.2 Text Vectorization (BOW and TF-IDF)
3.3 Models

3.3.1 Naive Bayes Models 

- Gaussian, Multinomial and Bernoulli Naive Bayes.
- Multinomial Naive Bayes along with TF-IDF gives best precision score and good accuracy.

3.3.2 Other Models

- Logistic Regression, Support Vector Classifier, Decision Tree Classifier, Random Forest Classifier, Random Forest Classifier, Bagging Classifier, 
- Extra Tree Classifier, AdaBoost Classifier, GradientBoost Classifier, K Neighbours Classifier, XGBoost Classifier. 
- Support Vector Classifier gives same precision score as that of Multinomial Naive Bayes

3.3.3 Experiments to increase Precision score
- TF-IDF with max_features hyper-parameter. Here accuracy score of best model increases with max precision score.
- Voting Classifier on top 3 Models. No significant change in precision score and accuracy score of best model.
- Using No. of characters column. Accuracy score of best model decreases.

3.3.4 Best model 
- Multinomial Naive Bayes along with TF-IDF and max_features equal to 3500.


### Deployment
- Web app is created for single prediction.
- Batch Prediction script is created for bulk prediction.


## Tech Stack Used
1. Python 
2. FlaskAPI 
3. Machine learning algorithms
4. HTML and CSS

## Infrastructure Required.
1. VS Code (you can use any other IDE)
2. GIT
3. GITHUB

## How to run?
This app is now available for offline run.
In coming days, i'll deploy this app on AWS.


### Step 1: Clone the repository
```bash
git clone https://github.com/Shubham3023/SmsEmailSpamClassifier.git
```

### Step 2- Download the code to local and use any IDE for running the code.

### Step 3 - Install the requirements
```bash
pip install -r requirements.txt
```


### Step 4.A - Run the application server for web app
```bash
1. Run app.py
```

```bash
2. Open this link in browser: http://127.0.0.1:5000
```

### Step 4.B. For batch prediction
```
1. Upload the input csv file to InputCSV directory.
2. Provide the  above file path to start_batch_prediction function.
3. Run batch_prediction.py file.
4. Prediction file is saved in Batch Prediction directory.


```

## Web Application
<img src="https://github.com/Shubham3023/SmsEmailSpamClassifier/blob/main/Notebook/application.png" alt="Web Application" />

## Spam Prediction
<img src="https://github.com/Shubham3023/SmsEmailSpamClassifier/blob/main/Notebook/Spam%20pred.jpg" alt="Spam Model Prediction" />

## Ham Prediction
<img src="https://github.com/Shubham3023/SmsEmailSpamClassifier/blob/main/Notebook/Ham%20Pred.jpg" alt="Ham Model Prediction" />
