import sys
import os
import pandas as pd
import numpy as np
from src.exception import CustomException
from src.logger import logging
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from dataclasses import dataclass
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from src.utils import save_obj

@dataclass
class DataTransmissionConfig:
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')

class DataTransmission:
    
    def __init__(self):
        self.data_transmission_config = DataTransmissionConfig()

    def data_transmission(self):
        try:
            num_features =  ['reading_score', 'writing_score']
            cat_features = ['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course']

            num_pipeline = Pipeline(
                steps = [
                    ("impute",SimpleImputer(strategy="median")),
                    ("standard",StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps = [
                    ("impute",SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder",OneHotEncoder())
                ]
            )

            logging.info(f"Categorical Feautres {cat_features}")
            logging.info(f"Numerical Features {num_features}")

            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline",num_pipeline,num_features),
                    ("cat_pipeline",cat_pipeline,cat_features)
                ]
            )
            logging.info("Preprocessor object created")

            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transmission(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Train and Test data Read")

            target_feature = "math_score"
            input_train_df = train_df.drop(columns=[target_feature],axis=1)
            target_train_df = train_df[target_feature]

            input_test_df = test_df.drop(columns=[target_feature],axis=1)
            target_test_df = test_df[target_feature]
            
            transformer = self.data_transmission()

            input_train_df_a = transformer.fit_transform(input_train_df)
            input_test_df_b = transformer.transform(input_test_df)

            test_arr = np.c_[input_test_df_b,np.array(target_test_df)] ## simply the input feature and output feature are concatinating
            train_arr = np.c_[input_train_df_a,np.array(target_train_df)]

            logging.info(f"Saving the Preprocessing objects")

            save_obj(
                file_path = self.data_transmission_config.preprocessor_obj_file_path,
                obj = transformer
            )
            logging.info("Preprocessor object Saved Successfully")
            return(
                train_arr,
                test_arr,
                self.data_transmission_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            raise CustomException(e,sys)
