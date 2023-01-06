from exception import CustomException
from logger import logging
import pandas as pd
from datetime import datetime
import os, sys
import joblib
from utils import text_transformer

### loading model and scalar object
logging.info("Reading model object from model.sav file")
model=joblib.load('Models\model.sav')
logging.info("Reading Vectorizer object from vectorizer.sav file")
vectorizer=joblib.load(r'Models\vectorizer.sav')


PREDICTION_DIR=os.path.join(os.getcwd(),"Batch_Prediction")


def start_batch_prediction(input_file_path):
    try:
        os.makedirs(PREDICTION_DIR,exist_ok=True)
        logging.info(f"Reading file :{input_file_path}")
        df = pd.read_csv(input_file_path, encoding ='latin1', names=['Target', 'Text']) 
        logging.info("Transforming the Input Text")
        df["Transformed_text"]=df['Text'].apply(text_transformer)
        logging.info("Vectorizing the Transformed text")
        X=vectorizer.fit_transform(df['Transformed_text']).toarray()
        logging.info("Using model to generate batch prediction")
        prediction=model.predict(X)  
        logging.info("Appending prediction {} to dataframe".format(prediction))     
        df["Prediction"]=prediction
        logging.info("Converting prediction values to categorical form")
        df["Prediction_Cat"]=df["Prediction"].astype(str).replace("1", "spam").replace("0", "ham")

        prediction_file_name = os.path.basename(input_file_path).replace(".csv",f"{datetime.now().strftime('%m%d%Y__%H%M%S')}.csv")
        prediction_file_path = os.path.join(PREDICTION_DIR,prediction_file_name)
        logging.info("Saving Prediction file to CSV")
        df.to_csv(prediction_file_path,index=False,header=True)
        logging.info("Batch Prediction Successful")
        print("Batch Prediction Successful")
    except Exception as e:
        raise CustomException(e, sys)


if __name__=="__main__":
    try:
        start_batch_prediction("Input_CSV\inspam.csv")
    except Exception as e:
        raise CustomException(e, sys)