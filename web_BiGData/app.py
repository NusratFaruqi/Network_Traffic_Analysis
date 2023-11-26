

import os

os.environ['PYSPARK_PYTHON'] = 'python'
from flask import Flask, request, render_template
import pandas as pd
from pyspark.ml.tuning import CrossValidatorModel
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import SparkSession





app = Flask(__name__,static_url_path='/static')





spark = SparkSession.builder.appName('TrafficAnalysis').config('spark.executor.memory', '2g').config('spark.executor.cores', '2').getOrCreate()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/dashboard', methods=['GET', 'POST'])
def dashboard():

 
    X=['FLOW_DURATION',
        'TOTAL_FWD_PACKETS',
        'TOTAL_BACKWARD_PACKETS',
        'TOTAL_LENGTH_OF_FWD_PACKETS',
        'TOTAL_LENGTH_OF_BWD_PACKETS',
        'FLOW_BYTES_S',
        'FLOW_PACKETS_S',
        'AVERAGE_PACKET_SIZE']

    if request.method == 'POST':
        try:
            
            model_path = os.path.abspath("network_traffic_analysis.model")
            model = CrossValidatorModel.load(model_path)
        

            input_data = [float(request.form[col]) for col in X]
           
            input_data = [tuple(input_data)]

          
            input_df = spark.createDataFrame(input_data, X)

          
            vec_asmbl = VectorAssembler(inputCols=X, outputCol='features')
            input_df = vec_asmbl.transform(input_df).select('features')
            
           
            predictions = model.transform(input_df)
            
           
            result = predictions.select("prediction").collect()[0]["prediction"]
            print(result)
        
    

            protocols=['TWITTER','INSTAGRAM','FACEBOOK']
            return render_template('prediction.html', prediction=protocols[int(result)])


        except Exception as e:
            print("ERROR----->:", str(e))



    return render_template('dashboard.html',X=X)

if __name__ == '__main__':
    app.run(debug=True)
