# Usar imagen base oficial de Python
FROM python:3.11-slim

# Establecer directorio de trabajo
WORKDIR /app

# Instalar dependencias del sistema para PDF processing
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copiar archivos de requirements primero para aprovechar cache de Docker
COPY requirements.txt .

# Instalar dependencias de Python
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código de la aplicación
COPY resumidor.py .
COPY app.py .
COPY static ./static

# Crear directorio para uploads temporales
RUN mkdir -p /tmp/uploads

# Exponer puerto
EXPOSE 8080

# Configurar variables de entorno
ENV PYTHONUNBUFFERED=1
ENV PORT=8080

# Comando para ejecutar la aplicación con gunicorn
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 app:app 