FROM python:3.7-slim-buster

RUN pip3 install boto3>=1.15.0 sagemaker pandas numpy s3fs
ENV PYTHONUNBUFFERED=TRUE

ENTRYPOINT [ "python3"]
