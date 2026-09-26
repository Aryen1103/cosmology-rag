output "url" {
  description = "Public URL of the app."
  value       = "https://${azurerm_container_app.main.ingress[0].fqdn}"
}

output "resource_group" {
  value = azurerm_resource_group.main.name
}
