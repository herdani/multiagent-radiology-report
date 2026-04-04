variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "eu-central-1"
}

variable "project_name" {
  description = "Project name"
  type        = string
  default     = "radiology-ai"
}

variable "db_username" {
  description = "PostgreSQL username"
  type        = string
  default     = "radiology"
  sensitive   = true
}

variable "db_password" {
  description = "PostgreSQL password"
  type        = string
  sensitive   = true
}

variable "groq_api_key" {
  description = "Groq API key"
  type        = string
  sensitive   = true
}

variable "openrouter_api_key" {
  description = "OpenRouter API key"
  type        = string
  sensitive   = true
}

variable "wandb_api_key" {
  description = "W&B API key"
  type        = string
  sensitive   = true
}

variable "container_port" {
  type    = number
  default = 8000
}

variable "ui_port" {
  type    = number
  default = 7860
}

variable "qdrant_url" {
  description = "Qdrant Cloud URL"
  type        = string
}

variable "qdrant_api_key" {
  description = "Qdrant Cloud API key"
  type        = string
  sensitive   = true
}
