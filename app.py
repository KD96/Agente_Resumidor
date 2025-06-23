#!/usr/bin/env python3
"""
Flask API para el resumidor de PDFs
Diseñado para ejecutarse en Google Cloud Run
"""

import os
import tempfile
import asyncio
from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename
from resumidor import PDFSummarizer
import logging
from pathlib import Path
import traceback

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configuración
UPLOAD_FOLDER = '/tmp/uploads'
ALLOWED_EXTENSIONS = {'pdf'}
MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50MB max file size

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Crear directorio de uploads si no existe
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    """Verificar si el archivo es un PDF válido"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_summarizer():
    """Crear instancia del resumidor con configuración de Cloud Run"""
    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    
    return PDFSummarizer(
        openai_api_key=api_key,
        model_name=os.environ.get('OPENAI_MODEL', 'gpt-3.5-turbo'),
        max_retries=int(os.environ.get('MAX_RETRIES', '3')),
        retry_delay=float(os.environ.get('RETRY_DELAY', '2.0'))
    )

@app.route('/', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'PDF Summarizer API',
        'version': '1.0.0'
    })

@app.route('/ui', methods=['GET'])
def ui():
    """Devuelve la interfaz web estática"""
    return app.send_static_file('index.html')

@app.route('/summarize', methods=['POST'])
def summarize_pdf():
    """
    Endpoint principal para resumir PDFs
    Acepta archivo PDF vía multipart/form-data
    """
    try:
        # Verificar que se envió un archivo
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        # Verificar que el archivo tiene nombre y es PDF
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Only PDF files are allowed'}), 400
        
        # Guardar el archivo temporalmente
        filename = secure_filename(file.filename)
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(temp_path)
        
        logger.info(f"Processing PDF: {filename}")
        
        # Inicializar el resumidor
        summarizer = get_summarizer()
        
        # Procesar el PDF
        result = summarizer.summarize_pdf(temp_path)
        
        # Limpiar archivo temporal
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        # Preparar respuesta
        response = {
            'success': True,
            'filename': filename,
            'summary': result['final_summary'],
            'metadata': {
                'total_pages': result['total_pages'],
                'document_count': result['document_count'],
                'source_file': filename
            }
        }
        
        # Incluir pasos intermedios si están disponibles
        if result.get('intermediate_steps'):
            response['intermediate_steps'] = result['intermediate_steps']
        
        logger.info(f"Successfully processed PDF: {filename}")
        return jsonify(response)
        
    except Exception as e:
        # Limpiar archivo temporal en caso de error
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.remove(temp_path)
        
        logger.error(f"Error processing PDF: {str(e)}")
        logger.error(traceback.format_exc())
        
        return jsonify({
            'success': False,
            'error': str(e),
            'message': 'Error processing PDF file'
        }), 500

@app.route('/batch-summarize', methods=['POST'])
def batch_summarize():
    """
    Endpoint para procesar múltiples PDFs
    Acepta múltiples archivos PDF
    """
    try:
        # Verificar que se enviaron archivos
        if 'files' not in request.files:
            return jsonify({'error': 'No files provided'}), 400
        
        files = request.files.getlist('files')
        
        if not files or all(f.filename == '' for f in files):
            return jsonify({'error': 'No files selected'}), 400
        
        # Filtrar solo archivos PDF válidos
        valid_files = [f for f in files if f.filename != '' and allowed_file(f.filename)]
        
        if not valid_files:
            return jsonify({'error': 'No valid PDF files provided'}), 400
        
        logger.info(f"Processing {len(valid_files)} PDF files")
        
        # Inicializar el resumidor
        summarizer = get_summarizer()
        
        results = {}
        temp_files = []
        
        for file in valid_files:
            try:
                # Guardar archivo temporal
                filename = secure_filename(file.filename)
                temp_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(temp_path)
                temp_files.append(temp_path)
                
                # Procesar PDF
                result = summarizer.summarize_pdf(temp_path)
                
                results[filename] = {
                    'success': True,
                    'summary': result['final_summary'],
                    'metadata': {
                        'total_pages': result['total_pages'],
                        'document_count': result['document_count']
                    }
                }
                
                logger.info(f"Successfully processed: {filename}")
                
            except Exception as e:
                logger.error(f"Error processing {filename}: {str(e)}")
                results[filename] = {
                    'success': False,
                    'error': str(e)
                }
        
        # Limpiar archivos temporales
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                os.remove(temp_file)
        
        return jsonify({
            'success': True,
            'results': results,
            'total_files': len(valid_files),
            'successful': len([r for r in results.values() if r.get('success')])
        })
        
    except Exception as e:
        # Limpiar archivos temporales en caso de error
        if 'temp_files' in locals():
            for temp_file in temp_files:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
        
        logger.error(f"Error in batch processing: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e),
            'message': 'Error in batch processing'
        }), 500

@app.route('/status', methods=['GET'])
def status():
    """Endpoint de estado del servicio"""
    try:
        # Verificar que tenemos acceso a OpenAI API
        api_key = os.environ.get('OPENAI_API_KEY')
        
        return jsonify({
            'status': 'operational',
            'openai_configured': bool(api_key),
            'upload_folder': app.config['UPLOAD_FOLDER'],
            'max_file_size_mb': app.config['MAX_CONTENT_LENGTH'] / (1024 * 1024),
            'supported_formats': list(ALLOWED_EXTENSIONS)
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

if __name__ == '__main__':
    # Para desarrollo local
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=False) 