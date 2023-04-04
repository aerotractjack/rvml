# import terraform AWS library
terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 4.0"
    }
  }
}

# setup AWS credentials
provider "aws" {
	shared_config_files      = ["/home/aerotract/.aws/config"]
	shared_credentials_files = ["/home/aerotract/.aws/credentials"]
	profile                  = "default"
}

# create bucket for client data
resource "aws_s3_bucket" "rvml-bucket" {
	bucket = "rvml-results"
}

# create and assign ACL to client bucket
resource "aws_s3_bucket_acl" "rvml-bucket-acl" {
	bucket = aws_s3_bucket.rvml-bucket.id
	acl    = "private"
}