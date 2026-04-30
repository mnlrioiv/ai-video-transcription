# Video Transcription & Summarization App

Aplicación web que permite subir videos, transcribir su audio usando Whisper (local) y generar resúmenes cortos y detallados usando la API de Google Gemini.

## Características

- Subida de videos mediante arrastre y suelte
- Transcripción de audio usando OpenAI Whisper (modelo base)
- Generación de resúmenes usando Google Gemini
- Interfaz web simple y responsiva

## Requisitos

- Python 3.8+
- FFmpeg instalado en el sistema
- Cuenta de Google Cloud con acceso a la API de Gemini
- Clave de API de Google guardada como variable de entorno `GOOGLE_API_KEY`

## Instalación

1. Clonar el repositorio
2. Instalar FFmpeg:
   ```bash
   # macOS
   brew install ffmpeg
   
   # Ubuntu/Debian
   sudo apt-get install ffmpeg
   
   # Windows
   # Descargar desde https://ffmpeg.org/download.html
   ```
3. Instalar dependencias de Python:
   ```bash
   cd backend
   pip install -r requirements.txt
   ```
4. Configurar la variable de entorno:
   ```bash
   export GOOGLE_API_KEY="tu_clave_de_api_aqui"
   ```
   O crear un archivo `.env` en el directorio backend:
   ```
   GOOGLE_API_KEY=tu_clave_de_api_aqui
   ```

## Ejecución

```bash
cd backend
uvicorn main:app --reload
```

Luego abrir en el navegador: http://localhost:8000

## Estructura del proyecto

```
video-transcription-app/
├── backend/
│   ├── main.py          # API FastAPI
│   ├── requirements.txt # Dependencias
│   └── static/
│       └── index.html   # Frontend simple
└── README.md
```

## Notas

- La aplicación usa Whisper "base" por defecto para un buen balance entre velocidad y precisión
- Para videos muy largos, considere usar un modelo más pequeño como "tiny" o implementar procesamiento por chunks
- La API de Gemini tiene límites de uso según el plan de Google Cloud