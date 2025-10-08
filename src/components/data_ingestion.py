import os
import pandas as pd
import sys
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from src.components.data_transform import DataTransmission

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts','train.csv')
    test_data_path: str = os.path.join('artifacts','test.csv')
    raw_data_path: str = os.path.join('artifacts','data.csv')

class DataIngestion():
    
    def __init__(self):
        self.ingestion_attribute = DataIngestionConfig()

    def initiate_data_ingestion(self):
        try:
            df = pd.read_csv(r'notebook\data\stud.csv')
            logging.info('Read the data')
            os.makedirs(os.path.dirname(self.ingestion_attribute.train_data_path),exist_ok=True) #here we are simply creating artifacts directory that's it instead of os.path.dirname we can directly give the 'artifacts'
            df.to_csv(self.ingestion_attribute.raw_data_path,index=False,header=True)

            logging.info('Performing the train test split')
            train_data,test_data = train_test_split(df,test_size=0.2,random_state=10)
            

            train_data.to_csv(self.ingestion_attribute.train_data_path,index = False,header = True)

            test_data.to_csv(self.ingestion_attribute.test_data_path, index = False,header = True)

            return(
                self.ingestion_attribute.train_data_path,
                self.ingestion_attribute.test_data_path
            )

        except Exception as e:
            raise CustomException(e,sys)
    
if __name__ == "__main__":
    obj = DataIngestion()
    train_path,test_path = obj.initiate_data_ingestion()
    obj2 = DataTransmission()
    obj2.initiate_data_transmission(train_path,test_path)
