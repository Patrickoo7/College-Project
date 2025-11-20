# Azure Deployment Guide

This guide provides step-by-step instructions for deploying the Heart Disease Prediction System to Microsoft Azure.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Quick Deployment](#quick-deployment)
- [Manual Deployment](#manual-deployment)
- [Configuration](#configuration)
- [Monitoring](#monitoring)
- [Scaling](#scaling)
- [Cost Optimization](#cost-optimization)
- [Troubleshooting](#troubleshooting)

## Prerequisites

### 1. Azure Account
- Active Azure subscription ([Free trial available](https://azure.microsoft.com/free/))
- Sufficient quota for:
  - App Service Plan (B1 or higher)
  - Azure Container Registry
  - Azure Storage (optional)

### 2. Tools Installation

**Azure CLI:**
```bash
# macOS
brew update && brew install azure-cli

# Windows (PowerShell as Administrator)
Invoke-WebRequest -Uri https://aka.ms/installazurecliwindows -OutFile .\AzureCLI.msi
Start-Process msiexec.exe -Wait -ArgumentList '/I AzureCLI.msi /quiet'

# Linux (Ubuntu/Debian)
curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash
```

**Docker:**
- Install from [docker.com](https://www.docker.com/get-started)
- Ensure Docker daemon is running

### 3. Login to Azure

```bash
az login

# Verify login
az account show

# Set subscription (if you have multiple)
az account set --subscription "YOUR_SUBSCRIPTION_ID"
```

## Quick Deployment

### Option 1: Automated Deployment Script

```bash
# Make script executable
chmod +x azure/deploy-to-azure.sh

# Run deployment
./azure/deploy-to-azure.sh
```

The script will:
1. ✅ Create Azure Resource Group
2. ✅ Set up Azure Container Registry (ACR)
3. ✅ Build and push Docker images
4. ✅ Create App Service Plan
5. ✅ Deploy API and Web applications
6. ✅ Configure environment variables

**Deployment time:** ~10-15 minutes

### Option 2: One-Click Deploy (Coming Soon)

[![Deploy to Azure](https://aka.ms/deploytoazurebutton)](https://portal.azure.com/#create/Microsoft.Template)

## Manual Deployment

### Step 1: Set Environment Variables

```bash
export RESOURCE_GROUP="heart-disease-rg"
export LOCATION="eastus"
export ACR_NAME="heartdiseaseacr"
export APP_SERVICE_PLAN="heart-disease-plan"
export API_APP_NAME="heart-disease-api"
export WEB_APP_NAME="heart-disease-web"
```

### Step 2: Create Resource Group

```bash
az group create \
    --name $RESOURCE_GROUP \
    --location $LOCATION
```

### Step 3: Create Azure Container Registry

```bash
# Create ACR
az acr create \
    --resource-group $RESOURCE_GROUP \
    --name $ACR_NAME \
    --sku Basic \
    --admin-enabled true

# Get ACR credentials
ACR_USERNAME=$(az acr credential show --name $ACR_NAME --query "username" -o tsv)
ACR_PASSWORD=$(az acr credential show --name $ACR_NAME --query "passwords[0].value" -o tsv)
ACR_LOGIN_SERVER="${ACR_NAME}.azurecr.io"

# Login to ACR
echo $ACR_PASSWORD | docker login $ACR_LOGIN_SERVER --username $ACR_USERNAME --password-stdin
```

### Step 4: Build and Push Docker Images

```bash
# Build API image
docker build -t ${ACR_LOGIN_SERVER}/heart-disease-api:latest -f docker/Dockerfile.api .

# Build Web image
docker build -t ${ACR_LOGIN_SERVER}/heart-disease-web:latest -f docker/Dockerfile.web .

# Push images
docker push ${ACR_LOGIN_SERVER}/heart-disease-api:latest
docker push ${ACR_LOGIN_SERVER}/heart-disease-web:latest
```

### Step 5: Create App Service Plan

```bash
az appservice plan create \
    --name $APP_SERVICE_PLAN \
    --resource-group $RESOURCE_GROUP \
    --is-linux \
    --sku B1  # Basic tier (upgrade for production)
```

**Recommended SKUs:**
- **B1**: Basic, good for development/testing ($13/month)
- **P1V2**: Production, better performance ($73/month)
- **P1V3**: Premium, with more CPU/RAM ($146/month)

### Step 6: Deploy API Application

```bash
# Create Web App for API
az webapp create \
    --resource-group $RESOURCE_GROUP \
    --plan $APP_SERVICE_PLAN \
    --name $API_APP_NAME \
    --deployment-container-image-name ${ACR_LOGIN_SERVER}/heart-disease-api:latest

# Configure container settings
az webapp config container set \
    --name $API_APP_NAME \
    --resource-group $RESOURCE_GROUP \
    --docker-custom-image-name ${ACR_LOGIN_SERVER}/heart-disease-api:latest \
    --docker-registry-server-url https://${ACR_LOGIN_SERVER} \
    --docker-registry-server-user $ACR_USERNAME \
    --docker-registry-server-password $ACR_PASSWORD

# Set environment variables
az webapp config appsettings set \
    --name $API_APP_NAME \
    --resource-group $RESOURCE_GROUP \
    --settings WEBSITES_PORT=8000 PYTHONUNBUFFERED=1
```

### Step 7: Deploy Web Application

```bash
# Create Web App for Streamlit
az webapp create \
    --resource-group $RESOURCE_GROUP \
    --plan $APP_SERVICE_PLAN \
    --name $WEB_APP_NAME \
    --deployment-container-image-name ${ACR_LOGIN_SERVER}/heart-disease-web:latest

# Configure container settings
az webapp config container set \
    --name $WEB_APP_NAME \
    --resource-group $RESOURCE_GROUP \
    --docker-custom-image-name ${ACR_LOGIN_SERVER}/heart-disease-web:latest \
    --docker-registry-server-url https://${ACR_LOGIN_SERVER} \
    --docker-registry-server-user $ACR_USERNAME \
    --docker-registry-server-password $ACR_PASSWORD

# Set environment variables
az webapp config appsettings set \
    --name $WEB_APP_NAME \
    --resource-group $RESOURCE_GROUP \
    --settings WEBSITES_PORT=8501 PYTHONUNBUFFERED=1
```

### Step 8: Verify Deployment

```bash
# Get application URLs
API_URL=$(az webapp show --name $API_APP_NAME --resource-group $RESOURCE_GROUP --query "defaultHostName" -o tsv)
WEB_URL=$(az webapp show --name $WEB_APP_NAME --resource-group $RESOURCE_GROUP --query "defaultHostName" -o tsv)

echo "API URL: https://${API_URL}"
echo "API Docs: https://${API_URL}/docs"
echo "Web App URL: https://${WEB_URL}"

# Test API health
curl https://${API_URL}/health
```

## Configuration

### Environment Variables

Set application-specific environment variables:

```bash
az webapp config appsettings set \
    --name $API_APP_NAME \
    --resource-group $RESOURCE_GROUP \
    --settings \
        WEBSITES_PORT=8000 \
        PYTHONUNBUFFERED=1 \
        LOG_LEVEL=INFO \
        GPU_ENABLED=false
```

### Custom Domain

1. **Add custom domain:**
```bash
az webapp config hostname add \
    --webapp-name $API_APP_NAME \
    --resource-group $RESOURCE_GROUP \
    --hostname "api.yourdomain.com"
```

2. **Enable HTTPS:**
```bash
az webapp config ssl bind \
    --name $API_APP_NAME \
    --resource-group $RESOURCE_GROUP \
    --certificate-thumbprint <thumbprint> \
    --ssl-type SNI
```

### Continuous Deployment

Enable continuous deployment from ACR:

```bash
az webapp deployment container config \
    --name $API_APP_NAME \
    --resource-group $RESOURCE_GROUP \
    --enable-cd true
```

Get webhook URL for CI/CD:
```bash
az webapp deployment container show-cd-url \
    --name $API_APP_NAME \
    --resource-group $RESOURCE_GROUP
```

## Monitoring

### Application Insights

1. **Create Application Insights:**
```bash
az monitor app-insights component create \
    --app heart-disease-insights \
    --location $LOCATION \
    --resource-group $RESOURCE_GROUP \
    --application-type web
```

2. **Get instrumentation key:**
```bash
INSTRUMENTATION_KEY=$(az monitor app-insights component show \
    --app heart-disease-insights \
    --resource-group $RESOURCE_GROUP \
    --query instrumentationKey -o tsv)
```

3. **Configure Web App:**
```bash
az webapp config appsettings set \
    --name $API_APP_NAME \
    --resource-group $RESOURCE_GROUP \
    --settings APPINSIGHTS_INSTRUMENTATIONKEY=$INSTRUMENTATION_KEY
```

### View Logs

```bash
# Stream logs
az webapp log tail \
    --name $API_APP_NAME \
    --resource-group $RESOURCE_GROUP

# Download logs
az webapp log download \
    --name $API_APP_NAME \
    --resource-group $RESOURCE_GROUP \
    --log-file logs.zip
```

## Scaling

### Manual Scaling

Scale out (add instances):
```bash
az appservice plan update \
    --name $APP_SERVICE_PLAN \
    --resource-group $RESOURCE_GROUP \
    --number-of-workers 3
```

Scale up (bigger instances):
```bash
az appservice plan update \
    --name $APP_SERVICE_PLAN \
    --resource-group $RESOURCE_GROUP \
    --sku P1V2
```

### Auto-Scaling

Enable autoscale based on CPU:
```bash
az monitor autoscale create \
    --resource-group $RESOURCE_GROUP \
    --resource $APP_SERVICE_PLAN \
    --resource-type Microsoft.Web/serverFarms \
    --name autoscale-plan \
    --min-count 1 \
    --max-count 5 \
    --count 1

az monitor autoscale rule create \
    --resource-group $RESOURCE_GROUP \
    --autoscale-name autoscale-plan \
    --condition "Percentage CPU > 70 avg 5m" \
    --scale out 1
```

## Cost Optimization

### 1. Right-size Resources

- **Development**: B1 tier ($13/month)
- **Production**: P1V2 tier ($73/month)
- **High-traffic**: P2V2 or P3V2

### 2. Use Deployment Slots

For staging/production:
```bash
az webapp deployment slot create \
    --name $API_APP_NAME \
    --resource-group $RESOURCE_GROUP \
    --slot staging
```

### 3. Monitor Costs

```bash
# View cost analysis
az consumption usage list \
    --start-date 2024-01-01 \
    --end-date 2024-01-31
```

### 4. Set Budget Alerts

```bash
az consumption budget create \
    --budget-name heart-disease-budget \
    --amount 100 \
    --time-grain Monthly \
    --time-period start=2024-01-01 end=2024-12-31
```

## Troubleshooting

### Common Issues

**Issue 1: Container fails to start**
```bash
# Check container logs
az webapp log tail --name $API_APP_NAME --resource-group $RESOURCE_GROUP

# Check container settings
az webapp config container show --name $API_APP_NAME --resource-group $RESOURCE_GROUP
```

**Issue 2: 502 Bad Gateway**
- Check if WEBSITES_PORT matches container port
- Verify health endpoint is responding
- Check application logs

**Issue 3: Authentication errors**
```bash
# Refresh ACR credentials
az acr credential renew --name $ACR_NAME --password-name password
```

### Debugging Commands

```bash
# Restart application
az webapp restart --name $API_APP_NAME --resource-group $RESOURCE_GROUP

# SSH into container
az webapp ssh --name $API_APP_NAME --resource-group $RESOURCE_GROUP

# Check resource usage
az monitor metrics list \
    --resource $API_APP_NAME \
    --resource-group $RESOURCE_GROUP \
    --resource-type "Microsoft.Web/sites" \
    --metric "CpuTime"
```

## Cleanup

To delete all resources and stop incurring charges:

```bash
# Delete entire resource group
az group delete --name $RESOURCE_GROUP --yes --no-wait

# Or delete specific resources
az webapp delete --name $API_APP_NAME --resource-group $RESOURCE_GROUP
az webapp delete --name $WEB_APP_NAME --resource-group $RESOURCE_GROUP
az acr delete --name $ACR_NAME --resource-group $RESOURCE_GROUP
az appservice plan delete --name $APP_SERVICE_PLAN --resource-group $RESOURCE_GROUP
```

## Additional Resources

- [Azure App Service Documentation](https://docs.microsoft.com/azure/app-service/)
- [Azure Container Registry](https://docs.microsoft.com/azure/container-registry/)
- [Azure Pricing Calculator](https://azure.microsoft.com/pricing/calculator/)
- [Azure Well-Architected Framework](https://docs.microsoft.com/azure/architecture/framework/)

## Support

For issues with:
- **Azure deployment**: Check Azure documentation or create support ticket
- **Application code**: Open issue on GitHub
- **Configuration**: Refer to configs/ directory documentation

---

**Happy deploying! 🚀**
