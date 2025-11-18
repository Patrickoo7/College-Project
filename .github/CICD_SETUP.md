# CI/CD Pipeline Setup Guide

This guide explains how to set up the GitHub Actions CI/CD pipeline for automated deployment to Azure.

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Setting Up Azure Credentials](#setting-up-azure-credentials)
- [Configuring GitHub Secrets](#configuring-github-secrets)
- [Pipeline Workflow](#pipeline-workflow)
- [Triggering Deployments](#triggering-deployments)
- [Monitoring Deployments](#monitoring-deployments)
- [Troubleshooting](#troubleshooting)

## Overview

The CI/CD pipeline automatically:

1. **Runs tests** on every push and pull request
2. **Checks code quality** with linting tools
3. **Builds Docker images** for API and Web applications
4. **Deploys to Azure** on pushes to main/master branch
5. **Performs health checks** after deployment

## Prerequisites

Before setting up the pipeline, ensure you have:

- [x] Azure account with active subscription
- [x] Azure resources created (Resource Group, ACR, App Services)
- [x] Azure CLI installed locally
- [x] GitHub repository with admin access
- [x] Completed initial Azure deployment using `azure/deploy-to-azure.sh`

## Setting Up Azure Credentials

### Step 1: Create Azure Service Principal

Create a service principal for GitHub Actions to authenticate with Azure:

```bash
# Set your subscription ID
SUBSCRIPTION_ID=$(az account show --query id -o tsv)

# Create service principal
az ad sp create-for-rbac \
  --name "github-actions-heart-disease" \
  --role contributor \
  --scopes /subscriptions/$SUBSCRIPTION_ID/resourceGroups/heart-disease-rg \
  --sdk-auth
```

This command will output JSON credentials. **Save this output securely** - you'll need it for the next step.

The output will look like:

```json
{
  "clientId": "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
  "clientSecret": "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
  "subscriptionId": "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
  "tenantId": "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
  "activeDirectoryEndpointUrl": "https://login.microsoftonline.com",
  "resourceManagerEndpointUrl": "https://management.azure.com/",
  "activeDirectoryGraphResourceId": "https://graph.windows.net/",
  "sqlManagementEndpointUrl": "https://management.core.windows.net:8443/",
  "galleryEndpointUrl": "https://gallery.azure.com/",
  "managementEndpointUrl": "https://management.core.windows.net/"
}
```

### Step 2: Grant ACR Permissions

Grant the service principal permission to push/pull from Azure Container Registry:

```bash
# Get ACR resource ID
ACR_ID=$(az acr show --name heartdiseaseacr --resource-group heart-disease-rg --query id -o tsv)

# Get service principal ID
SP_ID=$(az ad sp list --display-name github-actions-heart-disease --query "[0].id" -o tsv)

# Assign AcrPush role
az role assignment create \
  --assignee $SP_ID \
  --role AcrPush \
  --scope $ACR_ID
```

## Configuring GitHub Secrets

### Step 1: Navigate to Repository Settings

1. Go to your GitHub repository
2. Click **Settings** → **Secrets and variables** → **Actions**
3. Click **New repository secret**

### Step 2: Add Required Secrets

Add the following secret:

#### AZURE_CREDENTIALS

- **Name**: `AZURE_CREDENTIALS`
- **Value**: The entire JSON output from the service principal creation (Step 1 above)

Example:
```json
{
  "clientId": "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
  "clientSecret": "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
  "subscriptionId": "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
  "tenantId": "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
  ...
}
```

### Step 3: Verify Environment Variables

The workflow uses these environment variables (already configured in `.github/workflows/azure-deploy.yml`):

```yaml
env:
  AZURE_RESOURCE_GROUP: heart-disease-rg
  AZURE_LOCATION: eastus
  ACR_NAME: heartdiseaseacr
  API_APP_NAME: heart-disease-api
  WEB_APP_NAME: heart-disease-web
  PYTHON_VERSION: '3.9'
```

**If you used different names during Azure deployment**, update these values in the workflow file.

## Pipeline Workflow

### Workflow Structure

The pipeline consists of multiple jobs:

```
┌─────────────┐
│   Push to   │
│   main/PR   │
└──────┬──────┘
       │
       ├──────────────┬──────────────┐
       │              │              │
┌──────▼──────┐ ┌────▼─────┐ ┌─────▼──────┐
│    Test     │ │   Lint   │ │   Build    │
│             │ │          │ │   (PR)     │
└──────┬──────┘ └────┬─────┘ └────────────┘
       │              │
       └──────┬───────┘
              │
     ┌────────▼─────────┐
     │  Build & Deploy  │
     │   (main only)    │
     └──────────────────┘
```

### Job Descriptions

1. **Test Job**:
   - Runs pytest tests
   - Generates coverage reports
   - Uploads to Codecov (optional)

2. **Lint Job**:
   - Runs flake8 for code quality
   - Checks black formatting
   - Validates import ordering with isort

3. **Build-and-Deploy Job** (main branch only):
   - Logs into Azure
   - Builds Docker images
   - Pushes to Azure Container Registry
   - Deploys to Azure Web Apps
   - Performs health checks

4. **Build-Only Job** (pull requests):
   - Builds Docker images locally
   - Validates build process
   - Does not deploy

## Triggering Deployments

### Automatic Deployment

Deployments are triggered automatically when:

1. **Push to main/master branch**:
   ```bash
   git push origin main
   ```

2. **Merge a pull request** to main/master

### Manual Deployment

You can also trigger deployments manually:

1. Go to **Actions** tab in GitHub
2. Select **Azure Deployment CI/CD** workflow
3. Click **Run workflow**
4. Select branch and click **Run workflow**

### Deployment Conditions

| Event | Test | Lint | Build | Deploy |
|-------|------|------|-------|--------|
| Push to main | ✅ | ✅ | ✅ | ✅ |
| Pull Request | ✅ | ✅ | ✅ | ❌ |
| Manual trigger | ✅ | ✅ | ✅ | ✅ |

## Monitoring Deployments

### GitHub Actions UI

1. Go to **Actions** tab in your repository
2. Click on the running workflow
3. View real-time logs for each job

### Deployment Summary

After successful deployment, a summary is generated showing:

- ✅ Deployment status
- 🔗 Application URLs (API, API Docs, Web App)
- 🐳 Docker image tags

### Application Logs

View application logs in Azure:

```bash
# API logs
az webapp log tail --name heart-disease-api --resource-group heart-disease-rg

# Web app logs
az webapp log tail --name heart-disease-web --resource-group heart-disease-rg
```

### Health Checks

The pipeline automatically checks API health after deployment:

```bash
curl https://heart-disease-api.azurewebsites.net/health
```

## Troubleshooting

### Common Issues

#### 1. Authentication Failed

**Error**: "Azure login failed"

**Solution**:
- Verify `AZURE_CREDENTIALS` secret is correctly set
- Ensure service principal has contributor role
- Check if service principal credentials haven't expired

```bash
# Recreate service principal
az ad sp create-for-rbac --name "github-actions-heart-disease" --role contributor --scopes /subscriptions/$SUBSCRIPTION_ID/resourceGroups/heart-disease-rg --sdk-auth
```

#### 2. ACR Access Denied

**Error**: "unauthorized: authentication required"

**Solution**:
- Grant ACR permissions to service principal

```bash
ACR_ID=$(az acr show --name heartdiseaseacr --resource-group heart-disease-rg --query id -o tsv)
SP_ID=$(az ad sp list --display-name github-actions-heart-disease --query "[0].id" -o tsv)
az role assignment create --assignee $SP_ID --role AcrPush --scope $ACR_ID
```

#### 3. Docker Build Failed

**Error**: "Error building Docker image"

**Solution**:
- Check Dockerfile syntax
- Verify all required files exist
- Test build locally:

```bash
docker build -t test-api -f docker/Dockerfile.api .
docker build -t test-web -f docker/Dockerfile.web .
```

#### 4. Deployment Timeout

**Error**: "Deployment timed out"

**Solution**:
- Check App Service plan has sufficient resources
- View application logs for errors
- Increase timeout in workflow (default: 10 minutes)

#### 5. Health Check Failed

**Error**: "API health check failed"

**Solution**:
- Wait longer for container startup (increase sleep time)
- Check application logs
- Verify port configuration (WEBSITES_PORT=8000)

```bash
az webapp log tail --name heart-disease-api --resource-group heart-disease-rg
```

### Debugging Commands

```bash
# Check workflow status
gh run list --workflow=azure-deploy.yml

# View workflow logs
gh run view <run-id> --log

# Check Azure Web App status
az webapp show --name heart-disease-api --resource-group heart-disease-rg --query state

# View container logs
az webapp log download --name heart-disease-api --resource-group heart-disease-rg --log-file logs.zip

# Restart applications
az webapp restart --name heart-disease-api --resource-group heart-disease-rg
az webapp restart --name heart-disease-web --resource-group heart-disease-rg
```

## Pipeline Configuration

### Customizing Workflow

To customize the workflow, edit `.github/workflows/azure-deploy.yml`:

**Change deployment branch**:
```yaml
on:
  push:
    branches:
      - main
      - develop  # Add custom branch
```

**Adjust Python version**:
```yaml
env:
  PYTHON_VERSION: '3.10'  # Change version
```

**Add more tests**:
```yaml
- name: Run integration tests
  run: pytest tests/integration/ -v
```

### Skipping CI

To skip CI on a commit:

```bash
git commit -m "docs: update README [skip ci]"
```

## Best Practices

1. **Always test locally first**:
   ```bash
   make test
   docker-compose up
   ```

2. **Use pull requests** for code review before merging to main

3. **Monitor deployments** in GitHub Actions and Azure Portal

4. **Set up notifications**:
   - Configure GitHub notifications for workflow failures
   - Set up Azure alerts for application errors

5. **Keep secrets secure**:
   - Never commit credentials to repository
   - Rotate service principal credentials periodically

6. **Review costs**:
   - Monitor Azure usage to avoid unexpected charges
   - Use appropriate App Service plan tier

## Additional Resources

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Azure CLI Reference](https://docs.microsoft.com/en-us/cli/azure/)
- [Azure App Service Deployment](https://docs.microsoft.com/en-us/azure/app-service/deploy-github-actions)
- [Docker Build Best Practices](https://docs.docker.com/develop/dev-best-practices/)

## Support

For issues with:
- **GitHub Actions**: Check workflow logs and GitHub Actions documentation
- **Azure Deployment**: Review Azure documentation or create support ticket
- **Application Code**: Open issue on GitHub repository

---

**Happy deploying! 🚀**
