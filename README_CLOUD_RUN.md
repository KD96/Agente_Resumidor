# PDF Summarizer - Despliegue en Google Cloud Run

Este documento explica cómo desplegar el resumidor de PDFs en Google Cloud Run.

## 📋 Prerrequisitos

1. **Google Cloud CLI**: Instala y configura gcloud
   ```bash
   # Instalar gcloud (macOS)
   brew install google-cloud-sdk
   
   # Autenticar
   gcloud auth login
   
   # Configurar proyecto
   gcloud config set project YOUR_PROJECT_ID
   ```

2. **API Key de OpenAI**: Necesitarás una clave API válida de OpenAI

3. **Docker** (opcional): Solo si quieres hacer build local

## 🚀 Despliegue Rápido

### Método 1: Script Automático

1. **Edita el PROJECT_ID** en `deploy.sh`:
   ```bash
   # Abre deploy.sh y cambia esta línea:
   PROJECT_ID="tu-project-id-aqui"
   ```

2. **Ejecuta el script**:
   ```bash
   chmod +x deploy.sh
   ./deploy.sh
   ```

3. **Configura la API Key de OpenAI**:
   ```bash
   gcloud run services update pdf-summarizer \
     --region us-central1 \
     --set-env-vars OPENAI_API_KEY=tu_api_key_aqui
   ```

### Método 2: Paso a Paso

1. **Habilitar APIs**:
   ```bash
   gcloud services enable cloudbuild.googleapis.com
   gcloud services enable run.googleapis.com
   gcloud services enable containerregistry.googleapis.com
   ```

2. **Construir imagen**:
   ```bash
   gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/pdf-summarizer .
   ```

3. **Desplegar**:
   ```bash
   gcloud run deploy pdf-summarizer \
     --image gcr.io/YOUR_PROJECT_ID/pdf-summarizer \
     --platform managed \
     --region us-central1 \
     --allow-unauthenticated \
     --memory 2Gi \
     --cpu 2 \
     --timeout 3600 \
     --set-env-vars OPENAI_API_KEY=tu_api_key_aqui
   ```

## 📊 Configuración

### Variables de Entorno

| Variable | Descripción | Valor por Defecto |
|----------|-------------|-------------------|
| `OPENAI_API_KEY` | **REQUERIDA** - Tu API key de OpenAI | - |
| `OPENAI_MODEL` | Modelo de OpenAI a usar | `gpt-3.5-turbo` |
| `MAX_RETRIES` | Máximo número de reintentos | `3` |
| `RETRY_DELAY` | Delay base entre reintentos (segundos) | `2.0` |

### Recursos Cloud Run

- **Memoria**: 2 GiB (recomendado para PDFs grandes)
- **CPU**: 2 vCPUs
- **Timeout**: 3600 segundos (1 hora)
- **Concurrencia**: 10 requests por instancia
- **Máximo instancias**: 100

## 🔗 Endpoints de la API

Una vez desplegado, tu servicio tendrá estos endpoints:

### GET `/`
Health check básico
```bash
curl https://your-service-url.run.app/
```

### GET `/status`
Estado del servicio y configuración
```bash
curl https://your-service-url.run.app/status
```

### POST `/summarize`
Resumir un solo PDF
```bash
curl -X POST \
  -F "file=@documento.pdf" \
  https://your-service-url.run.app/summarize
```

### POST `/batch-summarize`
Resumir múltiples PDFs
```bash
curl -X POST \
  -F "files=@doc1.pdf" \
  -F "files=@doc2.pdf" \
  https://your-service-url.run.app/batch-summarize
```

## 🧪 Ejemplos de Uso

### Python
```python
import requests

# URL de tu servicio
url = "https://your-service-url.run.app/summarize"

# Subir PDF
with open("documento.pdf", "rb") as f:
    files = {"file": f}
    response = requests.post(url, files=files)
    
if response.status_code == 200:
    result = response.json()
    print("Resumen:", result["summary"])
else:
    print("Error:", response.json())
```

### JavaScript
```javascript
const formData = new FormData();
formData.append('file', fileInput.files[0]);

fetch('https://your-service-url.run.app/summarize', {
    method: 'POST',
    body: formData
})
.then(response => response.json())
.then(data => {
    if (data.success) {
        console.log('Resumen:', data.summary);
    } else {
        console.error('Error:', data.error);
    }
});
```

## 💰 Costos Estimados

### Google Cloud Run
- **CPU**: ~$0.000024 por vCPU-segundo
- **Memory**: ~$0.0000025 por GiB-segundo
- **Requests**: $0.40 por millón de requests

### OpenAI API (GPT-3.5-turbo)
- **Input**: $0.0015 per 1K tokens
- **Output**: $0.002 per 1K tokens

**Ejemplo**: PDF de 100 páginas (~150K tokens)
- Cloud Run: ~$0.02 por procesamiento
- OpenAI: ~$0.50 por resumen
- **Total**: ~$0.52 por PDF

## 🔧 Solución de Problemas

### Error: "OPENAI_API_KEY not set"
```bash
gcloud run services update pdf-summarizer \
  --region us-central1 \
  --set-env-vars OPENAI_API_KEY=tu_api_key_aqui
```

### Error: "Rate limit exceeded"
- Verifica que tu cuenta OpenAI tenga créditos
- Aumenta `RETRY_DELAY` si es necesario
- Considera usar `gpt-4` para menos requests

### Timeouts en PDFs grandes
```bash
gcloud run services update pdf-summarizer \
  --region us-central1 \
  --timeout 3600 \
  --memory 4Gi
```

### Ver logs
```bash
gcloud logs read --service pdf-summarizer --limit 50
```

## 🔒 Seguridad

### Para uso en producción:
1. **Remueve** `--allow-unauthenticated`
2. **Configura** IAM apropiado
3. **Usa** Cloud Secret Manager para API keys:
   ```bash
   # Crear secreto
   echo "tu_api_key" | gcloud secrets create openai-api-key --data-file=-
   
   # Actualizar servicio
   gcloud run services update pdf-summarizer \
     --region us-central1 \
     --set-env-vars OPENAI_API_KEY=projects/YOUR_PROJECT/secrets/openai-api-key/versions/latest
   ```

## 📈 Monitoreo

### Métricas en Cloud Console
- Request count
- Response time
- Error rate
- Memory utilization
- CPU utilization

### Alertas recomendadas
- Error rate > 5%
- Response time > 30s
- Memory utilization > 80%

## 🔄 Actualizaciones

Para actualizar el servicio:
```bash
# Re-ejecutar el script de despliegue
./deploy.sh

# O manualmente
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/pdf-summarizer .
gcloud run deploy pdf-summarizer --image gcr.io/YOUR_PROJECT_ID/pdf-summarizer
``` 