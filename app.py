import sys
from flask import Flask,render_template,request
application = Flask(__name__)
app = application
from src.pipeline.predict_pipeline import CustomData,PredictPipeline
from src.logger import logging
from src.exception import CustomException

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict',methods=["GET","POST"])
def predicting():
    if request.method == "GET":
        return render_template("home.html")
    else:
        try:
            pipeline = CustomData(
                gender = request.form.get('gender'),
                race_ethnicity = request.form.get('ethnicity'),
                parental_level_of_education = request.form.get('parent_education'),
                lunch = request.form.get('lunch'),
                test_preparation_course= request.form.get('test_course'),
                reading_score = request.form.get('reading_score'),
                writing_score = request.form.get('writing_score')
            )
            df = pipeline.convert_data_to_dataframe()
            logging.info(df)
            print(df)
            predict_pipeline_obj = PredictPipeline()
            result = predict_pipeline_obj.predict_res(df)
            return render_template('home.html',results=result[0])
        
        except Exception as e:
            raise CustomException(e,sys)
        
if __name__ == "__main__":
    app.run(debug=True)