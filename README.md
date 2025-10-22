# AI Map Open Data Demo

A web application that visualizes data on an interactive map using Flask and Leaflet.js.

## Setup

1. Install Python requirements:
```
pip install -r requirements.txt
```

2. Run the application:
```
python app.py
```

3. Open your browser and navigate to `http://localhost:5000`

## Project Structure

```
├── app.py              # Flask application
├── requirements.txt    # Python dependencies
├── static/            
│   ├── css/           # Stylesheets
│   │   └── style.css
│   └── js/            # JavaScript files
│       └── map.js
└── templates/         # HTML templates
    └── index.html
```

## Optional: Use a Hugging Face hosted model (free/community models)

You can configure the app to call a Hugging Face Inference API model (for example community-hosted Gemma/Gemini-like models) by setting two environment variables before starting the server:

- `HF_API_KEY` — your Hugging Face API key (https://huggingface.co/settings/tokens). Some community models allow unauthenticated access but a key is recommended.
- `HF_MODEL` — the model id (for example `google/flan-t5-small` or a community model id).

Example (PowerShell):
```powershell
$env:HF_API_KEY = 'hf_...'  # set your key
$env:HF_MODEL = 'google/flan-t5-small'
python app.py
```

If these variables are set, the server will attempt to call the HF model for responses. If not set (or on failure), the app uses a local lightweight fallback responder or ChatterBot if available.