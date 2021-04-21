# PM2.5 Forecast Model Deployment with TensorFlow Serving
PM2.5 forecast model has been trained by using ConvLSTM

## Let's get started

### Train a model and save it
Run ```ConvLSTM.py```.

### Customize the output of a SavedModel
If you run ```custom_ConvLSTM.py```, it will load a model which has been saved by ```ConvLSTM.py``` and 
convert the output as PM2.5 labels, indicating the status. (very bad, bad, good, and so on) 

### Configuration for models
Modify ```./models/pm25/models.config``` according to your environment.

### Start TensorFlow Serving server
~~~
./serving_run.sh
~~~

### Check the predictions
You can receive the predictions by running ```serving_test.py```.