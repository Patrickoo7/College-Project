#!/bin/bash

# Azure Deployment Script for Heart Disease Prediction System
# This script deploys the application to Azure Container Registry and Azure App Service

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration (set these as environment variables or update here)
RESOURCE_GROUP="${AZURE_RESOURCE_GROUP:-heart-disease-rg}"
LOCATION="${AZURE_LOCATION:-eastus}"
ACR_NAME="${AZURE_ACR_NAME:-heartdiseaseacr}"
APP_SERVICE_PLAN="${AZURE_APP_PLAN:-heart-disease-plan}"
API_APP_NAME="${AZURE_API_APP:-heart-disease-api}"
WEB_APP_NAME="${AZURE_WEB_APP:-heart-disease-web}"

echo -e "${GREEN}=====================================${NC}"
echo -e "${GREEN}Heart Disease Prediction - Azure Deployment${NC}"
echo -e "${GREEN}=====================================${NC}"

# Check if Azure CLI is installed
if ! command -v az &> /dev/null; then
    echo -e "${RED}Error: Azure CLI is not installed${NC}"
    echo "Install it from: https://docs.microsoft.com/en-us/cli/azure/install-azure-cli"
    exit 1
fi

# Login to Azure (if not already logged in)
echo -e "${YELLOW}Checking Azure login status...${NC}"
az account show &> /dev/null || az login

# Create Resource Group
echo -e "${YELLOW}Creating resource group: ${RESOURCE_GROUP}${NC}"
az group create --name $RESOURCE_GROUP --location $LOCATION

# Create Azure Container Registry
echo -e "${YELLOW}Creating Azure Container Registry: ${ACR_NAME}${NC}"
az acr create \
    --resource-group $RESOURCE_GROUP \
    --name $ACR_NAME \
    --sku Basic \
    --admin-enabled true

# Get ACR login credentials
echo -e "${YELLOW}Getting ACR credentials...${NC}"
ACR_USERNAME=$(az acr credential show --name $ACR_NAME --query "username" -o tsv)
ACR_PASSWORD=$(az acr credential show --name $ACR_NAME --query "passwords[0].value" -o tsv)
ACR_LOGIN_SERVER="${ACR_NAME}.azurecr.io"

# Login to ACR
echo -e "${YELLOW}Logging in to Azure Container Registry...${NC}"
echo $ACR_PASSWORD | docker login $ACR_LOGIN_SERVER --username $ACR_USERNAME --password-stdin

# Build and push API image
echo -e "${YELLOW}Building and pushing API Docker image...${NC}"
docker build -t ${ACR_LOGIN_SERVER}/heart-disease-api:latest -f docker/Dockerfile.api .
docker push ${ACR_LOGIN_SERVER}/heart-disease-api:latest

# Build and push Web image
echo -e "${YELLOW}Building and pushing Web Docker image...${NC}"
docker build -t ${ACR_LOGIN_SERVER}/heart-disease-web:latest -f docker/Dockerfile.web .
docker push ${ACR_LOGIN_SERVER}/heart-disease-web:latest

# Create App Service Plan
echo -e "${YELLOW}Creating App Service Plan: ${APP_SERVICE_PLAN}${NC}"
az appservice plan create \
    --name $APP_SERVICE_PLAN \
    --resource-group $RESOURCE_GROUP \
    --is-linux \
    --sku B1  # Basic tier, upgrade to P1V2 or higher for production

# Create API Web App
echo -e "${YELLOW}Creating API Web App: ${API_APP_NAME}${NC}"
az webapp create \
    --resource-group $RESOURCE_GROUP \
    --plan $APP_SERVICE_PLAN \
    --name $API_APP_NAME \
    --deployment-container-image-name ${ACR_LOGIN_SERVER}/heart-disease-api:latest

# Configure API Web App
az webapp config container set \
    --name $API_APP_NAME \
    --resource-group $RESOURCE_GROUP \
    --docker-custom-image-name ${ACR_LOGIN_SERVER}/heart-disease-api:latest \
    --docker-registry-server-url https://${ACR_LOGIN_SERVER} \
    --docker-registry-server-user $ACR_USERNAME \
    --docker-registry-server-password $ACR_PASSWORD

# Set API environment variables
az webapp config appsettings set \
    --name $API_APP_NAME \
    --resource-group $RESOURCE_GROUP \
    --settings \
        WEBSITES_PORT=8000 \
        PYTHONUNBUFFERED=1

# Create Web App (Streamlit)
echo -e "${YELLOW}Creating Web App: ${WEB_APP_NAME}${NC}"
az webapp create \
    --resource-group $RESOURCE_GROUP \
    --plan $APP_SERVICE_PLAN \
    --name $WEB_APP_NAME \
    --deployment-container-image-name ${ACR_LOGIN_SERVER}/heart-disease-web:latest

# Configure Web App
az webapp config container set \
    --name $WEB_APP_NAME \
    --resource-group $RESOURCE_GROUP \
    --docker-custom-image-name ${ACR_LOGIN_SERVER}/heart-disease-web:latest \
    --docker-registry-server-url https://${ACR_LOGIN_SERVER} \
    --docker-registry-server-user $ACR_USERNAME \
    --docker-registry-server-password $ACR_PASSWORD

# Set Web environment variables
az webapp config appsettings set \
    --name $WEB_APP_NAME \
    --resource-group $RESOURCE_GROUP \
    --settings \
        WEBSITES_PORT=8501 \
        PYTHONUNBUFFERED=1

# Get Web App URLs
API_URL=$(az webapp show --name $API_APP_NAME --resource-group $RESOURCE_GROUP --query "defaultHostName" -o tsv)
WEB_URL=$(az webapp show --name $WEB_APP_NAME --resource-group $RESOURCE_GROUP --query "defaultHostName" -o tsv)

echo -e "${GREEN}=====================================${NC}"
echo -e "${GREEN}Deployment Complete!${NC}"
echo -e "${GREEN}=====================================${NC}"
echo -e "${GREEN}API URL: https://${API_URL}${NC}"
echo -e "${GREEN}Web App URL: https://${WEB_URL}${NC}"
echo -e "${GREEN}API Docs: https://${API_URL}/docs${NC}"
echo -e "${GREEN}=====================================${NC}"

# Save deployment info
cat > azure/deployment-info.txt <<EOF
Deployment Information
=====================
Date: $(date)
Resource Group: $RESOURCE_GROUP
Location: $LOCATION
Container Registry: $ACR_NAME

API Application:
  Name: $API_APP_NAME
  URL: https://${API_URL}
  Docs: https://${API_URL}/docs

Web Application:
  Name: $WEB_APP_NAME
  URL: https://${WEB_URL}

To update the application:
1. Build new images: docker build ...
2. Push to ACR: docker push ...
3. Restart web apps:
   az webapp restart --name $API_APP_NAME --resource-group $RESOURCE_GROUP
   az webapp restart --name $WEB_APP_NAME --resource-group $RESOURCE_GROUP
EOF

echo -e "${YELLOW}Deployment info saved to azure/deployment-info.txt${NC}"
