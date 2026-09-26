terraform {
  required_version = ">= 1.9"

  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 4.0"
    }
  }

  # Remote state in an Azure Storage container; settings are passed at init time
  # (-backend-config), so nothing account-specific is committed. See README.md.
  backend "azurerm" {}
}

provider "azurerm" {
  features {}
  # Subscription and credentials come from ARM_* environment variables (OIDC in CI, az login locally).
}
