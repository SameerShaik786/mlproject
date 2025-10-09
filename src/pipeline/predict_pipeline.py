import sys
from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomException
from src.utils import save_obj,load_file
import pandas as pd

@dataclass
class PredictPipeline:
    def __init__ (self):
        pass

    def predict_res(self,features):
        try:
            model_path = 'artifacts/model.pkl'
            preprocessor_path = 'artifacts/preprocessor.pkl'
            model = load_file(file_name = model_path)
            preprocessor = load_file(file_name = preprocessor_path)
            resultant_features = preprocessor.transform(features)
            result = model.predict(resultant_features)
            return result
        except Exception as e:
            raise CustomException(e,sys)

class CustomData:
    def __init__(self,gender,race_ethnicity,parental_level_of_education,lunch,
                 test_preparation_course,reading_score,writing_score):
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score
    
    def convert_data_to_dataframe(self):
        try:
            data_dict = {
            "gender" : [self.gender],
            "race_ethnicity":[self.race_ethnicity],
            "parental_level_of_education" : [self.parental_level_of_education],
            "lunch" : [self.lunch],
            "test_preparation_course" : [self.test_preparation_course],
            "reading_score" : [self.reading_score],
            "writing_score" : [self.writing_score]
            }
            return pd.DataFrame(data_dict)
        except Exception as e:
            raise CustomException(e,sys)