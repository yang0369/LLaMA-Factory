# SageMaker Pipeline for LLaMA-Factory

## Build and push custom LLaMA-Factory docker

We edited llama-factory to be compatible with SageMaker pipelines. See [the main Python execution file](https://github.com/billcai/llama-factory-modified/blob/main/src/train_bash_custom.py).

You can build and push with the following commands:
```
AWS_ACCOUNT_ID=052567997892

aws ecr get-login-password --region ap-southeast-1 | docker login --username AWS --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.ap-southeast-1.amazonaws.com

docker build /home/kewen_yang/LLMOps/llama-factory-modified/ -t $AWS_ACCOUNT_ID.dkr.ecr.ap-southeast-1.amazonaws.com/ecr-aiss-sagemaker:llama2

docker push $AWS_ACCOUNT_ID.dkr.ecr.ap-southeast-1.amazonaws.com/ecr-aiss-sagemaker:llama2
```

## Create the pipeline
First, you need to create an IAM Role that can be assumed by SageMaker, with the following trust relationship:
```
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Principal": {
                "Service": "sagemaker.amazonaws.com"
            },
            "Action": "sts:AssumeRole"
        }
    ]
}
```

Then with the following IAM permissions:
```
AmazonSageMakerFullAccess
AmazonS3FullAccess
AmazonEC2ContainerRegistryFullAccess
```

Run pipeline.py.
```
AWS_ACCOUNT_ID=<account id>
ARTIFACT_BUCKET=<bucket to store artifacts>
ROLE_ARN=<IAM Role created above>
python pipeline.py
```