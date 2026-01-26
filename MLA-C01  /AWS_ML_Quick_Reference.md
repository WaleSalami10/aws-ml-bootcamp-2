# AWS ML Associate - Quick Reference Guide

## SageMaker Built-in Algorithms Cheat Sheet

### Supervised Learning Algorithms

| Algorithm | Problem Type | Use Cases | Key Hyperparameters |
|-----------|--------------|-----------|-------------------|
| **Linear Learner** | Classification/Regression | Linear relationships, baseline models | learning_rate, l1, l2, epochs |
| **XGBoost** | Classification/Regression | Tabular data, feature importance | max_depth, eta, subsample, num_round |
| **Factorization Machines** | Classification/Regression | Sparse data, recommendations | factors, lr, epochs |
| **k-NN** | Classification/Regression | Simple baseline, non-parametric | k, predictor_type |

### Unsupervised Learning Algorithms

| Algorithm | Problem Type | Use Cases | Key Hyperparameters |
|-----------|--------------|-----------|-------------------|
| **K-Means** | Clustering | Customer segmentation, data exploration | k, epochs, init_method |
| **PCA** | Dimensionality Reduction | Feature reduction, visualization | algorithm_mode, subtract_mean |
| **Random Cut Forest** | Anomaly Detection | Fraud detection, outlier identification | num_trees, num_samples_per_tree |

### Deep Learning Algorithms

| Algorithm | Problem Type | Use Cases | Key Hyperparameters |
|-----------|--------------|-----------|-------------------|
| **Image Classification** | Computer Vision | Image recognition, medical imaging | epochs, learning_rate, batch_size |
| **Object Detection** | Computer Vision | Object localization, autonomous vehicles | epochs, learning_rate, mini_batch_size |
| **Semantic Segmentation** | Computer Vision | Medical imaging, autonomous driving | epochs, learning_rate, batch_size |
| **BlazingText** | NLP | Text classification, word embeddings | mode, epochs, learning_rate |

---

## SageMaker Deployment Options

### Real-time Inference
```python
# Single model endpoint
predictor = model.deploy(
    initial_instance_count=1,
    instance_type='ml.m5.large',
    endpoint_name='my-endpoint'
)

# Multi-model endpoint
mme = MultiDataModel(
    name='my-multi-model',
    model_data_prefix='s3://bucket/models/',
    role=role
)
```

### Batch Transform
```python
transformer = model.transformer(
    instance_count=1,
    instance_type='ml.m5.large',
    output_path='s3://bucket/output/'
)
transformer.transform('s3://bucket/input/')
```

### Serverless Inference
```python
predictor = model.deploy(
    serverless_inference_config=ServerlessInferenceConfig(
        memory_size_in_mb=1024,
        max_concurrency=5
    )
)
```

---

## Model Evaluation Metrics

### Classification Metrics
```python
# Binary Classification
accuracy = (TP + TN) / (TP + TN + FP + FN)
precision = TP / (TP + FP)
recall = TP / (TP + FN)
f1_score = 2 * (precision * recall) / (precision + recall)

# Multi-class
macro_avg = sum(metric_per_class) / num_classes
micro_avg = sum(TP) / sum(TP + FP)  # for precision
```

### Regression Metrics
```python
import numpy as np

# Mean Absolute Error
mae = np.mean(np.abs(y_true - y_pred))

# Mean Squared Error
mse = np.mean((y_true - y_pred) ** 2)

# Root Mean Squared Error
rmse = np.sqrt(mse)

# R-squared
ss_res = np.sum((y_true - y_pred) ** 2)
ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
r2 = 1 - (ss_res / ss_tot)
```

---

## SageMaker Training Job Configuration

### Basic Training Job
```python
from sagemaker.estimator import Estimator

estimator = Estimator(
    image_uri='382416733822.dkr.ecr.us-east-1.amazonaws.com/xgboost:latest',
    role=role,
    instance_count=1,
    instance_type='ml.m5.large',
    output_path='s3://bucket/output/',
    hyperparameters={
        'max_depth': 5,
        'eta': 0.2,
        'objective': 'binary:logistic',
        'num_round': 100
    }
)

estimator.fit({'train': 's3://bucket/train/', 'validation': 's3://bucket/val/'})
```

### Hyperparameter Tuning
```python
from sagemaker.tuner import HyperparameterTuner, IntegerParameter, ContinuousParameter

hyperparameter_ranges = {
    'max_depth': IntegerParameter(3, 10),
    'eta': ContinuousParameter(0.1, 0.5),
    'subsample': ContinuousParameter(0.5, 1.0)
}

tuner = HyperparameterTuner(
    estimator=estimator,
    objective_metric_name='validation:auc',
    hyperparameter_ranges=hyperparameter_ranges,
    max_jobs=20,
    max_parallel_jobs=3
)

tuner.fit({'train': train_input, 'validation': validation_input})
```

---

## Data Preparation Patterns

### Feature Engineering with SageMaker Processing
```python
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.sklearn.processing import SKLearnProcessor

processor = SKLearnProcessor(
    framework_version='0.23-1',
    role=role,
    instance_type='ml.m5.large',
    instance_count=1
)

processor.run(
    code='preprocessing.py',
    inputs=[ProcessingInput(
        source='s3://bucket/raw-data/',
        destination='/opt/ml/processing/input'
    )],
    outputs=[ProcessingOutput(
        source='/opt/ml/processing/output',
        destination='s3://bucket/processed-data/'
    )]
)
```

### Data Validation with Deequ
```python
# Example preprocessing.py for data quality checks
import pandas as pd
from pydeequ import Check, CheckLevel, ConstraintSuggestionRunner, VerificationSuite

df = pd.read_csv('/opt/ml/processing/input/data.csv')

# Data quality checks
check = Check(spark, CheckLevel.Warning, "Data Quality Check")
check.hasSize(lambda x: x >= 1000)  # Minimum rows
check.isComplete("target_column")   # No missing values
check.isUnique("id_column")        # Unique identifiers

verification_result = VerificationSuite(spark).onData(df).addCheck(check).run()
```

---

## Monitoring and Alerting

### Model Monitor Setup
```python
from sagemaker.model_monitor import DefaultModelMonitor
from sagemaker.model_monitor.dataset_format import DatasetFormat

monitor = DefaultModelMonitor(
    role=role,
    instance_count=1,
    instance_type='ml.m5.large',
    volume_size_in_gb=20,
    max_runtime_in_seconds=3600
)

# Create baseline
monitor.suggest_baseline(
    baseline_dataset='s3://bucket/baseline/data.csv',
    dataset_format=DatasetFormat.csv(header=True),
    output_s3_uri='s3://bucket/baseline-results'
)

# Schedule monitoring
monitor.create_monitoring_schedule(
    monitor_schedule_name='my-model-monitor',
    endpoint_input=predictor.endpoint_name,
    output_s3_uri='s3://bucket/monitoring-results',
    statistics=monitor.baseline_statistics(),
    constraints=monitor.suggested_constraints(),
    schedule_cron_expression='cron(0 * * * ? *)'  # Hourly
)
```

### CloudWatch Metrics and Alarms
```python
import boto3

cloudwatch = boto3.client('cloudwatch')

# Create custom metric
cloudwatch.put_metric_data(
    Namespace='SageMaker/ModelMetrics',
    MetricData=[
        {
            'MetricName': 'ModelAccuracy',
            'Value': accuracy_score,
            'Unit': 'Percent',
            'Dimensions': [
                {
                    'Name': 'EndpointName',
                    'Value': endpoint_name
                }
            ]
        }
    ]
)

# Create alarm
cloudwatch.put_metric_alarm(
    AlarmName='ModelAccuracyAlarm',
    ComparisonOperator='LessThanThreshold',
    EvaluationPeriods=2,
    MetricName='ModelAccuracy',
    Namespace='SageMaker/ModelMetrics',
    Period=300,
    Statistic='Average',
    Threshold=80.0,
    ActionsEnabled=True,
    AlarmActions=['arn:aws:sns:region:account:topic-name']
)
```

---

## Security Best Practices

### IAM Role for SageMaker
```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "s3:GetObject",
                "s3:PutObject",
                "s3:DeleteObject",
                "s3:ListBucket"
            ],
            "Resource": [
                "arn:aws:s3:::my-ml-bucket",
                "arn:aws:s3:::my-ml-bucket/*"
            ]
        },
        {
            "Effect": "Allow",
            "Action": [
                "logs:CreateLogGroup",
                "logs:CreateLogStream",
                "logs:PutLogEvents"
            ],
            "Resource": "arn:aws:logs:*:*:*"
        }
    ]
}
```

### VPC Configuration
```python
from sagemaker.vpc_utils import VpcConfig

vpc_config = VpcConfig(
    security_group_ids=['sg-12345678'],
    subnets=['subnet-12345678', 'subnet-87654321']
)

estimator = Estimator(
    # ... other parameters
    subnets=vpc_config.subnets,
    security_group_ids=vpc_config.security_group_ids
)
```

### Encryption Configuration
```python
from sagemaker.s3 import S3Uploader
from sagemaker import get_execution_role

# S3 encryption
S3Uploader.upload(
    local_path='data.csv',
    desired_s3_uri='s3://bucket/encrypted-data/',
    kms_key='arn:aws:kms:region:account:key/key-id'
)

# Training job encryption
estimator = Estimator(
    # ... other parameters
    encrypt_inter_container_traffic=True,
    volume_kms_key='arn:aws:kms:region:account:key/key-id',
    output_kms_key='arn:aws:kms:region:account:key/key-id'
)
```

---

## Cost Optimization Strategies

### Spot Instances for Training
```python
estimator = Estimator(
    # ... other parameters
    use_spot_instances=True,
    max_wait=7200,  # 2 hours
    max_run=3600,   # 1 hour
    checkpoint_s3_uri='s3://bucket/checkpoints/'
)
```

### Multi-Model Endpoints
```python
from sagemaker.multidatamodel import MultiDataModel

# Create multi-model endpoint
mme = MultiDataModel(
    name='multi-model-endpoint',
    model_data_prefix='s3://bucket/models/',
    role=role,
    image_uri=image_uri
)

predictor = mme.deploy(
    initial_instance_count=1,
    instance_type='ml.m5.large'
)

# Add/remove models dynamically
mme.add_model(model_data_source='s3://bucket/models/model1.tar.gz', model_data_path='model1')
mme.delete_model('model1')
```

### Automatic Scaling
```python
import boto3

autoscaling = boto3.client('application-autoscaling')

# Register scalable target
autoscaling.register_scalable_target(
    ServiceNamespace='sagemaker',
    ResourceId=f'endpoint/{endpoint_name}/variant/AllTraffic',
    ScalableDimension='sagemaker:variant:DesiredInstanceCount',
    MinCapacity=1,
    MaxCapacity=10
)

# Create scaling policy
autoscaling.put_scaling_policy(
    PolicyName='SageMakerVariantInvocationsPerInstance',
    ServiceNamespace='sagemaker',
    ResourceId=f'endpoint/{endpoint_name}/variant/AllTraffic',
    ScalableDimension='sagemaker:variant:DesiredInstanceCount',
    PolicyType='TargetTrackingScaling',
    TargetTrackingScalingPolicyConfiguration={
        'TargetValue': 70.0,
        'PredefinedMetricSpecification': {
            'PredefinedMetricType': 'SageMakerVariantInvocationsPerInstance'
        }
    }
)
```

---

## Common Exam Scenarios

### Scenario 1: Real-time Inference with Low Latency
**Solution:** Use real-time endpoints with appropriate instance types (compute-optimized for CPU, GPU instances for deep learning)

### Scenario 2: Batch Processing Large Datasets
**Solution:** Use SageMaker Batch Transform with multiple instances and appropriate data splitting

### Scenario 3: Cost-Effective Model Hosting
**Solution:** Multi-model endpoints for multiple models, serverless inference for variable traffic

### Scenario 4: Model Performance Degradation
**Solution:** SageMaker Model Monitor with CloudWatch alarms and automated retraining pipelines

### Scenario 5: Data Privacy and Compliance
**Solution:** VPC configuration, encryption at rest and in transit, IAM policies, and audit logging

Remember: Always choose AWS-native, managed services when possible, and consider cost, security, and scalability in your solutions!