sudo docker run --rm -t -p 8501:8501 --name tf_serving_pm25_forecast -v ${PWD}/models:/models tensorflow/serving --model_config_file=/models/models.config --model_config_file_poll_wait_seconds=30
