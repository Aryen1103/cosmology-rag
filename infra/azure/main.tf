locals {
  use_registry_auth = var.registry_server != "" && var.registry_username != ""

  # Only keys that were provided become secrets; a missing key just disables that provider in the app.
  api_key_secrets = {
    for env_name, value in {
      DEEPSEEK_API_KEY  = var.deepseek_api_key
      ANTHROPIC_API_KEY = var.anthropic_api_key
    } : env_name => value if value != ""
  }
}

resource "azurerm_resource_group" "main" {
  name     = "rg-${var.name}"
  location = var.location
  tags     = var.tags
}

resource "azurerm_log_analytics_workspace" "main" {
  name                = "log-${var.name}"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  sku                 = "PerGB2018"
  retention_in_days   = 30
  daily_quota_gb      = 0.5 # hard cap so a noisy loop can't run up a logging bill
  tags                = var.tags
}

# Consumption-only environment: pay per use, scales to zero.
resource "azurerm_container_app_environment" "main" {
  name                       = "cae-${var.name}"
  location                   = azurerm_resource_group.main.location
  resource_group_name        = azurerm_resource_group.main.name
  log_analytics_workspace_id = azurerm_log_analytics_workspace.main.id
  tags                       = var.tags
}

resource "azurerm_container_app" "main" {
  name                         = "ca-${var.name}"
  container_app_environment_id = azurerm_container_app_environment.main.id
  resource_group_name          = azurerm_resource_group.main.name
  revision_mode                = "Single"
  tags                         = var.tags

  dynamic "secret" {
    for_each = local.api_key_secrets
    content {
      name  = lower(replace(secret.key, "_", "-"))
      value = secret.value
    }
  }

  dynamic "secret" {
    for_each = local.use_registry_auth ? [1] : []
    content {
      name  = "registry-password"
      value = var.registry_password
    }
  }

  dynamic "registry" {
    for_each = local.use_registry_auth ? [1] : []
    content {
      server               = var.registry_server
      username             = var.registry_username
      password_secret_name = "registry-password"
    }
  }

  ingress {
    external_enabled = true
    target_port      = 8000
    transport        = "auto"

    traffic_weight {
      latest_revision = true
      percentage      = 100
    }
  }

  template {
    min_replicas = var.min_replicas
    max_replicas = 1 # the question limits are in-memory, so they only hold with one replica

    container {
      name   = "cosmology-rag"
      image  = var.image
      cpu    = 1.0
      memory = "2Gi" # the app idles at ~400 MiB; headroom for torch during queries

      dynamic "env" {
        for_each = local.api_key_secrets
        content {
          name        = env.key
          secret_name = lower(replace(env.key, "_", "-"))
        }
      }

      env {
        name  = "ASK_LIMIT_PER_IP_PER_HOUR"
        value = tostring(var.ask_limit_per_ip_per_hour)
      }

      env {
        name  = "ASK_LIMIT_PER_DAY"
        value = tostring(var.ask_limit_per_day)
      }

      # Loading the index and embedding model takes several seconds after a cold start.
      startup_probe {
        transport               = "HTTP"
        port                    = 8000
        path                    = "/api/health"
        interval_seconds        = 5
        failure_count_threshold = 24
      }

      liveness_probe {
        transport = "HTTP"
        port      = 8000
        path      = "/api/health"
      }
    }
  }
}
