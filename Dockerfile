FROM python:3.8
WORKDIR /app
COPY SVM_DecisionTree_train.py /app/
COPY requirements.txt /app/
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
RUN pip install -r requirements.txt
VOLUME /app/saved_model
CMD ["python", "SVM_DecisionTree_train.py"]