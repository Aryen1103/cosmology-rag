variable "name" {
  description = "Base name for all resources."
  type        = string
  default     = "cosmology-rag"
}

variable "location" {
  description = "Azure region."
  type        = string
  default     = "westeurope"
}

variable "image" {
  description = "Container image to run, e.g. ghcr.io/<owner>/cosmology-rag:<git-sha>. Built locally, because the corpus in data/ is not in the repo."
  type        = string
}

variable "registry_server" {
  description = "Registry host for a private image (e.g. ghcr.io). Leave empty for a public image."
  type        = string
  default     = ""
}

variable "registry_username" {
  type    = string
  default = ""
}

variable "registry_password" {
  description = "Registry token with read access (e.g. a GitHub PAT with read:packages)."
  type        = string
  default     = ""
  sensitive   = true
}

variable "deepseek_api_key" {
  description = "DeepSeek API key. Stored as a Container Apps secret, never in the image."
  type        = string
  default     = ""
  sensitive   = true
}

variable "anthropic_api_key" {
  description = "Anthropic API key (optional)."
  type        = string
  default     = ""
  sensitive   = true
}

variable "ask_limit_per_ip_per_hour" {
  description = "Questions per client per hour. Every question spends API credit."
  type        = number
  default     = 10
}

variable "ask_limit_per_day" {
  description = "Model calls per UTC day across all clients."
  type        = number
  default     = 200
}

variable "min_replicas" {
  description = "0 scales to zero when idle (near-free, ~30-60 s cold start); 1 keeps it warm (billed continuously)."
  type        = number
  default     = 0

  validation {
    condition     = var.min_replicas >= 0 && var.min_replicas <= 1
    error_message = "Use 0 or 1: the rate limiter is in-memory, so the app must run as a single replica."
  }
}

variable "tags" {
  type = map(string)
  default = {
    project = "cosmology-rag"
  }
}
