# Machine Learning Zoomcamp - Midterm Project 2023

## Project Title: Predicting Wine Quality

<p align="center">
  <img src="img/wine-quality-prediction.jpg" alt="Data Schema" width="300" height="300">
</p>

## Problem Statement

The task is to predict the quality of the wine (the target attribute) based on its physicochemical properties (the predictive features). The quality of the wine is a score between 0 and 10. This problem involves training regression ML models to predict a continuous target attribute (wine quality) from given features (physicochemical properties).

## Technologies used

- Cloud: AWS
- Cloud deployment service: Amazon Lightsail Containers
- EDA: Pandas profiling
- Feature importance: Shapash
- Model training & tuning: Scikit-Learn
- Model deployment: REST API (Flask)
- Containerization: Docker
- Dependency management: Pipenv
- Notebook server: JupyterLab 

## Dataset Description

The dataset consists of several attributes:

1. `fixed acidity`, `volatile acidity`, `citric acid`, `residual sugar`, `chlorides`, `free sulfur dioxide`, `total sulfur dioxide`, `density`, `pH`, `sulphates`, `alcohol`, `wine type`: 

These represent the physicochemical properties of the wine. They are the main predictive features for this problem.

2. `quality`

This is the quality score of the wine, which serves as the target attribute for the regression problem.

This dataset allows us to train a variety of regression algorithms and evaluate them to select the best model for deployment. 

This dataset can be accessed from the [HuggingFace Datasets library](https://huggingface.co/datasets/codesignal/wine-quality).

## Setting the Development Environment:

The developtment environment consists of two components:

- Python virtual environment: We set-up a `pipenv` enviornment consisting of a `Pipfile` and `Pipfile.lock` files. These document the Python dependencies required. 

```console
cd jupyter
pipenv lock
```

- Docker image: We build the `ml-zoom-mid-dev` docker image that encapsulates the pipenv environment and mounts the project root path. We then launch a container which will start a JupyterLab server for our development. 

```console
docker build -t ml-zoom-mid-dev .
docker run -it --rm -p 8888:8888 -p 5000:5000 -v ${PWD}:/app --name ml-zoom-mid-dev-env ml-zoom-mid-dev
```

- Open a browser window and navigate to http://localhost:8888/lab

## Data analysis & Model building resources

| File Name          | Directory                      | Description                                           |
|------------------------|------------------------------------|--------------------------------------------------|
| `01_data_etl.ipynb`        | eda  | Data ingestion from Hugging Face datasets. Data splitting. Data persistence.|
| `02_eda_profiling.ipynb`   | eda  | Exploratory data analysis. Data transformation. Feature engineering.        |
| `train_profile_report.html`   | eda  | Initial exploratory data report.         |
| `transformed_train_profile_report.html`   | eda  | Exploratory data report after data transformation.         |
| `03_feature_importance.ipynb`   | eda  | Feature importance. Feature-target correlation.        |
| `train.csv`, `validate.csv`, `test.csv`    | data  | Initial train, validation, and testing datasets.       |
| `train_transformed.csv`, `validate_transformed.csv`, `test_transformed.csv`   | data  | Train, validation, and testing datasets after data transformation.      |
| `04_model_training.ipynb`   | ml_train  | Data preparation. Model training. Hyperparameter tuning. Model evaluation. Model persistence.|
| `best_model.pkl`   | models  | The best regression model based on model evaluation.|
| `encoder.pkl`   | models  | One hot encoding model to encode the categorical data.|

## Model Deployment (Local)

The local model deployment consists of the below steps:

- Python virtual environment: We set-up a `pipenv` enviornment consisting of a `Pipfile` and `Pipfile.lock` files. These document the Python dependencies required. 

```console
cd ml_deploy
pipenv lock
```

- Flask app: We develop a [Flask script](ml_deploy/app.py) that uses the model to serve prediction requests through the `/predict` endpoint. 

```python
@app.route('/predict', methods=['POST'])
def predict_endpoint():
    obs = request.get_json()

    client_data = prepare_features(obs)
    pred = predict(client_data)

    data = {'score': round(pred[0], 2)}
    return jsonify(data)
```

- Docker image: We build the `ml-zoom-mid-dep` docker image that encapsulates the Flask app. We then launch a container which will mount the Flask port number (5000) and start Flask app for model serving using the gunicorn WSGI server. 

```console
docker build -t ml-zoom-mid-dep .
docker run -it -p 5000:5000 --name ml-zoom-mid-dep-env ml-zoom-mid-dep 
```

- Testing: We send a curl request with a sample record to the prediction endpoint for testing:

```console
curl http://127.0.0.1:5000/predict -d '{"alcohol": 9.4, "chlorides_density_ratio": 0.076, "citric acid": 0, "fixed acidity": 7.4, "pH": 3.51, "residual_sugar_density_mean": 1.9, "sulfur_dioxide_mean": 22.5, "sulphates": 0.56, "volatile acidity": 0.7, "wine_type": "red"}' -H 'Content-Type: application/json' 

output:
-------
{"score":6.9}
```

## Model Deployment (Cloud)

To deploy the model in the cloud, we apply the below steps to use [Amazon Lightsail](https://aws.amazon.com/lightsail/) to deploy the Flask app, which has been deployed locally, in a Docker container on AWS:

- Build a container using [Dockerfile](ml_deploy_cloud/Dockerfile)

```console
docker build -t flask-container-aws .
```

- Once the container build is done, test the Flask app locally by running the container and sending a curl request to the prediction endpoint with a sample record.

```console
docker build -t flask-container-aws .
docker run -p 5000:5000 flask-container-aws
docker exec -it 56593662b90f bash
curl http://127.0.0.1:5000/predict -d '{"alcohol": 9.4, "chlorides_density_ratio": 0.076, "citric acid": 0, "fixed acidity": 7.4, "pH": 3.51, "residual_sugar_density_mean": 1.9, "sulfur_dioxide_mean": 22.5, "sulphates": 0.56, "volatile acidity": 0.7, "wine_type": "red"}' -H 'Content-Type: application/json'

output:
-------
{"score":6.9}
```

- Install and configure Amazon Lightsail Control (lightsailctl) [plugin](https://lightsail.aws.amazon.com/ls/docs/en_us/articles/amazon-lightsail-install-software)

- Create a Lightsail container service.

```console
aws lightsail create-container-service --service-name flask-service-aws --power small --scale 1
```

- Use the `get-container-services` command to monitor the state of the container as it is being created. Wait until the container service state changes to “READY” before continuing to the next step.

```console
aws lightsail get-container-services

output:
-------
    "containerServices": [
        {
            ...
            "containerServiceName": "flask-service-aws",
            "state": "READY",
            ...
        }
    ]

```

- Push the application container to Amazon Lightsail.

```console
aws lightsail push-container-image --service-name flask-service-aws --label flask-container-aws --image flask-container-aws

output:
-------
Image "flask-container-aws" registered.
Refer to this image as ":flask-service-aws.flask-container-aws.1" in deployments.
```

- Create a new file, [containers.json](ml_deploy_cloud/containers.json) that describes the settings of the containers that will be launched on the container service.

- Create a new file, [public-endpoint.json](ml_deploy_cloud/public-endpoint.json) that describes the settings of the public endpoint for the container service.

- Deploy the container to the container service with the AWS CLI. The output of the create-container-servicedeployment command indicates that the state of the container service is now “DEPLOYING” as shown.

```console
aws lightsail create-container-service-deployment --service-name flask-service-aws --containers file://containers.json --public-endpoint file://public-endpoint.json

output:
-------
    "containerService": {
      ...
        "containerServiceName": "flask-service-aws",
        "state": "DEPLOYING"
      ...
    }
    
```

- Use the `get-container-services` command to monitor the state of the container until it changes to “RUNNING” before continuing to the next step.

```console
aws lightsail get-container-services --service-name flask-service-aws

output:
-------
    "containerService": {
      ...
        "containerServiceName": "flask-service-aws",
        "state": "RUNNING"
      ...
    }
    
```

- Test the Flask app remotely by sending a curl request to the prediction endpoint with a sample record.

```console
curl https://flask-service-aws.9lcraqcp6032e.us-east-1.cs.amazonlightsail.com/predict -d '{"alcohol": 9.4, "chlorides_density_ratio": 0.076, "citric acid": 0, "fixed acidity": 7.4, "pH": 3.51, "residual_sugar_density_mean": 1.9, "sulfur_dioxide_mean": 22.5, "sulphates": 0.56, "volatile acidity": 0.7, "wine_type": "red"}' -H 'Content-Type: application/json'

output:
-------
{"score":6.9}
```