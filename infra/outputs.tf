output "alb_url" {
  description = "Public URL for the demo — share this with recruiters"
  value       = "http://${aws_lb.main.dns_name}"
}

output "gradio_url" {
  description = "Gradio UI URL"
  value       = "http://${aws_lb.main.dns_name}"
}

output "api_url" {
  description = "FastAPI backend URL"
  value       = "http://${aws_lb.main.dns_name}/docs"
}

output "ecr_repository_url" {
  description = "ECR repository URL for CI/CD"
  value       = data.aws_ecr_repository.api.repository_url
}

output "s3_bucket" {
  description = "S3 bucket for DICOM images"
  value       = aws_s3_bucket.dicom_store.bucket
}

output "rds_endpoint" {
  description = "RDS endpoint"
  value       = aws_db_instance.postgres.endpoint
  sensitive   = true
}
