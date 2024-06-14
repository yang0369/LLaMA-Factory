import boto3
import sagemaker
import sagemaker.session
import os, json

from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput
from sagemaker.model_metrics import (
    MetricsSource,
    ModelMetrics,
)
from sagemaker.workflow.parameters import (
    ParameterString,
)
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import (
    TrainingStep,
)

from sagemaker.workflow.model_step import ModelStep
from sagemaker.model import Model
from sagemaker.workflow.pipeline_context import PipelineSession
import datetime

AWS_ACCOUNT_ID = '052567997892'
ARTIFACT_BUCKET = 's3://s3-imda-aipo-sagemaker-data/data'
ROLE_ARN = 'arn:aws:iam::052567997892:role/iamrole-aipo-sagemaker'

def get_sagemaker_client(region):
     """Gets the sagemaker client.

        Args:
            region: the aws region to start the session
            default_bucket: the bucket to use for storing the artifacts

        Returns:
            `sagemaker.session.Session instance
        """
     boto_session = boto3.Session(region_name=region)
     sagemaker_client = boto_session.client("sagemaker")
     return sagemaker_client

def get_sagemaker_client(region):
     """Gets the sagemaker client.

        Args:
            region: the aws region to start the session
            default_bucket: the bucket to use for storing the artifacts

        Returns:
            `sagemaker.session.Session instance
        """
     boto_session = boto3.Session(region_name=region)
     sagemaker_client = boto_session.client("sagemaker")
     return sagemaker_client


def get_session(region, default_bucket):
    """Gets the sagemaker session based on the region.
    Args:
        region: the aws region to start the session
        default_bucket: the bucket to use for storing the artifacts
    Returns:
        `sagemaker.session.Session instance
    """
    boto_session = boto3.Session(region_name=region)
    sagemaker_client = boto_session.client("sagemaker")
    runtime_client = boto_session.client("sagemaker-runtime")
    return sagemaker.session.Session(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        sagemaker_runtime_client=runtime_client,
        default_bucket=default_bucket,
    )

def get_pipeline_session(region, default_bucket):
    """Gets the pipeline session based on the region.
    Args:
        region: the aws region to start the session
        default_bucket: the bucket to use for storing the artifacts
    Returns:
        PipelineSession instance
    """
    boto_session = boto3.Session(region_name=region)
    sagemaker_client = boto_session.client("sagemaker")
    return PipelineSession(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        default_bucket=default_bucket,
    )

region = 'ap-southeast-1'
default_bucket = None
role = ROLE_ARN
sagemaker_session = get_session(region, default_bucket)
pipeline_session = get_pipeline_session(region, default_bucket)

model_approval_status = ParameterString(
        name="ModelApprovalStatus", default_value="PendingManualApproval"
    )
input_data = ParameterString(
        name="InputDataUrl",
        default_value="s3://s3-imda-aipo-sagemaker-data/data",
    )
num_gpus = ParameterString(
        name="NumGPUs",
        default_value="4",
    )
execution_command = ParameterString(
    name = "TrainingArguments",
    default_value="--stage sft --do_train --model_name_or_path Qwen/Qwen1.5-4B-Chat --dataset alpaca_gpt4_en --dataset_dir /opt/ml/input/data/training --template qwen --finetuning_type lora --lora_target q_proj,v_proj --output_dir /opt/ml/model --overwrite_cache --overwrite_output_dir --cutoff_len 1024"
)
exeuction_command_additional = ParameterString(
    name="AdditionalTrainingArguments",
    default_value="--per_device_train_batch_size 4 --per_device_eval_batch_size 4 --gradient_accumulation_steps 2 --lr_scheduler_type cosine --logging_steps 10 --save_steps 100 --eval_steps 100 --evaluation_strategy steps --load_best_model_at_end --learning_rate 5e-1 --num_train_epochs 1.0 --max_samples 30 --val_size 0.1 --plot_loss --fp16"
)
# image_uri = "052567997892.dkr.ecr.ap-southeast-1.amazonaws.com/ecr-aiss-sagemaker@sha256:15f414a1cdfa3b290c8c5608051d0d46ddbb971f5770e5f5fafb95326de03181"
image_uri = "052567997892.dkr.ecr.ap-southeast-1.amazonaws.com/ecr-aiss-sagemaker:llama2"
model_path = f"s3://s3-imda-aipo-sagemaker-data/finetuning_output/"
job_name= 'llm-finetune-{}'.format(
        datetime.datetime.now().isoformat()
    )
metrics_definitions = [
    {
        "Name": "train:loss",
        "Regex": "'loss': (.*?),"
    },
    {
        "Name": "eval:loss",
        "Regex": "'eval_loss': (.*?),"
    }
]
finetuning_instance_type = ParameterString(
        name="FinetuningInstanceType",
        default_value="ml.p4de.24xlarge",
    )
estimator = Estimator(
        image_uri=image_uri,
        instance_type=finetuning_instance_type,
        instance_count=1,
        output_path=model_path,
        sagemaker_session=pipeline_session,
        role=role,
        max_run=5*24*60*60,
        metric_definitions=metrics_definitions,
        container_arguments=['python', 'src/train_bash_custom.py'],
        keep_alive_period_in_seconds=300,
        environment={
            'SM_USE_RESERVED_CAPACITY': '1',
            'TRAIN_ARGS': execution_command,
            'TRAIN_ARGS_ADD': exeuction_command_additional,
            'NUM_GPUS': num_gpus
        }
    )
step_args = estimator.fit(
        job_name=job_name,
        inputs={
            "training": TrainingInput(
                s3_data=input_data
            ),
        },
    )
step_train = TrainingStep(
        name="FinetuneLLM",
        step_args=step_args,
    )
# this needs to change
# deploy_image_uri = f"{AWS_ACCOUNT_ID}.dkr.ecr.ap-southeast-1.amazonaws.com/ecr-aiss-sagemaker:latest"

deploy_image_uri = "052567997892.dkr.ecr.ap-southeast-1.amazonaws.com/ecr-aiss-sagemaker@sha256:15f414a1cdfa3b290c8c5608051d0d46ddbb971f5770e5f5fafb95326de03181"
model = Model(
        image_uri=deploy_image_uri,
        model_data=step_train.properties.ModelArtifacts.S3ModelArtifacts,
        sagemaker_session=pipeline_session,
        role=None
    )
step_args = model.register(
    content_types=["application/json"],
    response_types=["application/json"],
    inference_instances=["ml.g4dn.xlarge"],
    transform_instances=["ml.g4dn.xlarge"],
    model_package_group_name="FinetunedLLMModelPackage",
    approval_status=model_approval_status,
    customer_metadata_properties={
        'INSTANCE_TYPE': finetuning_instance_type,
        'IMAGE_URI': image_uri,
        'NUM_GPUS': num_gpus
    }
)
step_register = ModelStep(
    name="RegisterModel",
    step_args=step_args,
)
pipeline = Pipeline(
    name='02may',
    parameters=[
        finetuning_instance_type,
        model_approval_status,
        input_data,
        execution_command,
        exeuction_command_additional,
        num_gpus,
    ],
    steps=[step_train, step_register],
    sagemaker_session=pipeline_session,
)
pipeline.create(
    role_arn=role,
    # tags={
    #     "Key": 'sagemaker:domain-arn',
    #     "Value": 'arn:aws:sagemaker:ap-southeast-1:052567997892'
    #     }
    )


"""
# check the available instances in sg
https://aws.amazon.com/sagemaker/pricing/
"""
