# terraform-aws-paper2remarkable-lambda
A Terraform module that creates an AWS Lambda function for calling [paper2remarkable](https://github.com/GjjvdBurg/paper2remarkable).

The Lambda uses a custom Docker image (see [src/Dockerfile](src/Dockerfile)) which comes with paper2remarkable and its dependencies installed.

You can opt-in to receive emails if paper2remarkable fails to process a given input, or if the Lambda function itself times out.

## Set up
You need to first manually create the tokens for [rmapi](https://github.com/juruen/rmapi/), and then store these tokens in the parameter `/<prefix>/rmapi-config` in AWS Parameter Store using the following format:
```json
{"rmapi_device_token": "<token>", "rmapi_user_token": "<token>"}
```

You also need to build the Docker image and host it in ECR, e.g.:
```sh
$ docker build -t "<aws-account-id>.dkr.ecr.<aws-region>.amazonaws.com/<prefix>-paper2remarkable" ./src
$ aws ecr get-login-password --region <aws-region> | docker login --username AWS --password-stdin "<aws-account-id>.dkr.ecr.<aws-region>.amazonaws.com"
$ docker push "<aws-account-id>.dkr.ecr.<aws-region>.amazonaws.com/<prefix>-paper2remarkable"
```

## Usage
After setting up the credentials, you can run the Lambda as such:
```sh
aws lambda invoke \
  --function-name <prefix>-paper2remarkable \
  --payload '{"inputs": ["https://arxiv.org/abs/1406.2661"]}' \
  output.json
```

You can further integrate the Lambda function with AWS API Gateway to set up a REST API for the function -- allowing you to send files to your reMarkable from anywhere with an internet connection!

## TODOs
- Host the image in a public ECR repository.
- The Lambda somestimes failes due to an error `[Errno 2] No such file or directory ` -- seems to be related to reuse of Lambda execution context. Changing the directory to a known directory seems to fix this, but may need to research this further and see if it can be handled in a more robust manner.
- A normal run takes roughly 60 seconds.
  - May be an idea to use a Fargate task instead of a Lambda, but there's an inherent ~30 sec start-up time here.
- Docker image can probably be simplified and minimized (currently sits at 500 MB).
