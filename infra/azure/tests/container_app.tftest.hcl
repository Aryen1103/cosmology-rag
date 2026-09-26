# Plan-level tests against a mocked provider: no Azure account or credentials needed.
# Run: terraform init -backend=false && terraform test

mock_provider "azurerm" {}

variables {
  image            = "ghcr.io/example/cosmology-rag:test"
  deepseek_api_key = "not-a-real-key"
}

run "scales_to_zero_as_a_single_replica" {
  command = plan

  assert {
    condition     = azurerm_container_app.main.template[0].min_replicas == 0
    error_message = "Should scale to zero by default to stay near-free."
  }

  assert {
    condition     = azurerm_container_app.main.template[0].max_replicas == 1
    error_message = "The in-memory rate limiter needs a single replica."
  }

  assert {
    condition     = azurerm_container_app.main.ingress[0].target_port == 8000
    error_message = "Ingress must target the port the container listens on."
  }
}

run "api_key_is_a_secret_reference_not_a_plain_env_value" {
  command = plan

  assert {
    condition = anytrue([
      for e in azurerm_container_app.main.template[0].container[0].env :
      e.name == "DEEPSEEK_API_KEY" && e.secret_name == "deepseek-api-key" && e.value == null
    ])
    error_message = "DEEPSEEK_API_KEY must come from a Container Apps secret."
  }
}

run "rate_limits_are_configured" {
  command = plan

  assert {
    condition = alltrue([
      for name, value in { ASK_LIMIT_PER_IP_PER_HOUR = "10", ASK_LIMIT_PER_DAY = "200" } :
      anytrue([for e in azurerm_container_app.main.template[0].container[0].env : e.name == name && e.value == value])
    ])
    error_message = "Public deployments must set both question limits."
  }
}

run "missing_anthropic_key_adds_no_secret" {
  command = plan

  assert {
    condition = !anytrue([
      for e in azurerm_container_app.main.template[0].container[0].env : e.name == "ANTHROPIC_API_KEY"
    ])
    error_message = "An unset provider key should not create an env var."
  }
}

run "rejects_multiple_replicas" {
  command = plan

  variables {
    min_replicas = 2
  }

  expect_failures = [var.min_replicas]
}
