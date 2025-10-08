import sys
import os 
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import (AdaBoostRegressor,GradientBoostingRegressor)
from catboost import CatBoostRegressor 
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from dataclasses import dataclass 
from sklearn.metrics import r2_score
from src.utils import evaluation_modals,save_obj
from src.exception import CustomException
from src.logger import logging

@dataclass
class ModelTrainerConfig:
    model_trainer_file_path = os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    
    def initiate_models(self,train_arr,test_arr,preprocessor_path):
        try:
            models:dict = {
                "Linear Regression" : LinearRegression(),
                "Random Forest": RandomForestRegressor(),
                "Decision Tree" : DecisionTreeRegressor(),
                "XGBoost" : XGBRegressor(),
                "Adaboost" : AdaBoostRegressor(),
                "Gradient Boosting" : GradientBoostingRegressor(),
                "SVM" :SVR(),
                "K-Nearest Neighbours" : KNeighborsRegressor(),
                "Catboost" : CatBoostRegressor()
            }

            X_train = train_arr[:,:-1]
            X_test = test_arr[:,:-1]
            y_train = train_arr[:,-1]
            y_test = test_arr[:,-1]

            models_result:dict = evaluation_modals(X_train,X_test,y_train,y_test,models)

            best_model_score = max(sorted(models_result.values()))

            best_algo = ""
            for key in models_result.keys():
                if models_result[key] ==  best_model_score:
                    best_algo = key
            
            logging.info(best_algo)
            
            best_model = models[best_algo]


            save_obj(
                file_path= self.model_trainer_config.model_trainer_file_path,
                obj = best_model
            )

            if best_model_score < 0.6:
                raise CustomException("No model is performing better",sys)

            logging.info("Best Model found on both training dataset and testing dataset")

            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test,predicted)
            return r2_square
        
        except Exception as e:
            raise CustomException(e,sys)