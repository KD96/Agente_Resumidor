#!/bin/bash

# Script de despliegue para Google Cloud Run
# Asegúrate de tener configurado gcloud y Docker

set -e

# Configuración
PROJECT_ID="peerless-clock-371715"  # Tu PROJECT_ID correcto
SERVICE_NAME="pdf-summarizer"
REGION="us-central1"
IMAGE_NAME="gcr.io/$PROJECT_ID/$SERVICE_NAME"

echo "🚀 Desplegando PDF Summarizer a Google Cloud Run..."
echo "Project ID: $PROJECT_ID"
echo "Service Name: $SERVICE_NAME"
echo "Region: $REGION"

# Verificar que gcloud está configurado
echo "📡 Verificando configuración de gcloud..."
gcloud config get-value project

# Habilitar APIs necesarias
echo "🔧 Habilitando APIs necesarias..."
gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable containerregistry.googleapis.com

# Construir la imagen usando Cloud Build
echo "🏗️  Construyendo imagen con Cloud Build..."
gcloud builds submit --tag $IMAGE_NAME .

# Desplegar a Cloud Run
echo "🚀 Desplegando a Cloud Run..."
gcloud run deploy $SERVICE_NAME \
    --image $IMAGE_NAME \
    --platform managed \
    --region $REGION \
    --allow-unauthenticated \
    --memory 2Gi \
    --cpu 2 \
    --timeout 3600 \
    --concurrency 10 \
    --max-instances 100 \
    --set-env-vars "OPENAI_MODEL=gpt-3.5-turbo" \
    --set-env-vars "MAX_RETRIES=3" \
    --set-env-vars "RETRY_DELAY=2.0"

# Obtener URL del servicio
SERVICE_URL=$(gcloud run services describe $SERVICE_NAME --platform managed --region $REGION --format 'value(status.url)')

echo "✅ Despliegue completado!"
echo "🌐 URL del servicio: $SERVICE_URL"
echo ""
echo "📚 Endpoints disponibles:"
echo "  GET  $SERVICE_URL/ - Health check"
echo "  GET  $SERVICE_URL/status - Estado del servicio"
echo "  POST $SERVICE_URL/summarize - Resumir un PDF"
echo "  POST $SERVICE_URL/batch-summarize - Resumir múltiples PDFs"
echo ""
echo "⚠️  IMPORTANTE: No olvides configurar la variable de entorno OPENAI_API_KEY"
echo "   Ejecuta el siguiente comando para configurarla:"
echo "   gcloud run services update $SERVICE_NAME --region $REGION --set-env-vars OPENAI_API_KEY=tu_api_key_aqui"
echo ""
echo "📖 Para probar el servicio:"
echo "   curl -X POST -F 'file=@tu_archivo.pdf' $SERVICE_URL/summarize" 