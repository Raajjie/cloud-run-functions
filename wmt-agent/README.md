# WMT Agent - Water Meter Reading Processing API

A sophisticated Flask-based REST API that processes water meter readings from natural language text using AI-powered extraction, validation, and data cleaning. The system combines Google Gemini AI for intelligent text processing with rule-based validation and conflict resolution.

## 🚀 Features

- **AI-Powered Extraction**: Uses Google Gemini AI to extract and standardize unit readings from various text formats
- **Intelligent Validation**: Validates extracted readings for duplicates and conflicts using both rule-based and AI-enhanced logic
- **Data Cleaning**: Automatically removes duplicates and resolves conflicts
- **Structured JSON Output**: Returns clean, structured JSON data with unit and reading pairs
- **ReAct Reasoning Loop**: Implements a sophisticated reasoning loop with thoughts, actions, and observations
- **Cloud Run Ready**: Optimized for deployment on Google Cloud Run
- **Comprehensive Logging**: Detailed processing logs for debugging and monitoring

## 🏗️ Architecture

The system uses a multi-step processing pipeline:

1. **Input Standardization** (Gemini AI)
2. **Regex Extraction** (Pattern matching)
3. **Validation** (Duplicate/Conflict detection)
4. **Data Cleaning** (Remove duplicates/conflicts)
5. **JSON Processing** (Final structured output)

## 📋 Prerequisites

- Python 3.11+
- Google Cloud account with billing enabled
- Google Gemini API key
- Docker (for containerized deployment)

## 📚 API Documentation

### Base URL
- **Local:** `http://localhost:8080`
- **Cloud Run:** `https://wmt-agent-192363053306.asia-southeast1.run.app`

### Endpoints

#### 1. Health Check
```http
GET /
```

**Response:**
```json
{
  "message": "Welcome to WMT API!"
}
```

#### 2. Process Unit Readings
```http
POST /process
```

**Request Body:**
```json
{
  "text": "Unit 1A reads 123.45 cubic meter, Unit 2B reads 67 m3, Unit 3C reads 89.12 cubic meters"
}
```

**Response:**
```json
[
  {
    "unit": "1A",
    "reading": 123.45
  },
  {
    "unit": "2B", 
    "reading": 67.0
  },
  {
    "unit": "3C",
    "reading": 89.12
  }
]
```

#### 3. Update Process (by index)
```http
PUT /process/{index}
```

**Request Body:**
```json
{
  "text": "Unit 1A reads 150.25 cubic meter"
}
```

#### 4. Get All Readings
```http
GET /readings
```

#### 5. Get Specific Reading
```http
GET /readings/{index}
```

#### 6. Delete All Readings
```http
DELETE /readings
```

#### 7. Delete Specific Reading
```http
DELETE /readings/{index}
```

#### 8. Get Processing Metadata
```http
GET /readings/{kind}
```

Where `{kind}` can be:
- `duplicates` - Get all duplicate detection results
- `conflicts` - Get all conflict detection results  
- `logs` - Get all processing logs

#### 9. Get Specific Metadata Item
```http
GET /readings/{kind}/{index}
```

## 💡 Usage Examples

### Using curl

**Process readings:**
```bash
curl -X POST https://your-service-url/process \
  -H "Content-Type: application/json" \
  -d '{"text": "Unit 1A reads 123.45 cubic meter, Unit 2B reads 67 m3"}'
```

**Get all readings:**
```bash
curl https://your-service-url/readings
```

### Using Python

```python
import requests

# Process readings
response = requests.post(
    'https://your-service-url/process',
    json={'text': 'Unit 1A reads 123.45 cubic meter, Unit 2B reads 67 m3'}
)

readings = response.json()
print(readings)
```


## 📝 Supported Text Formats

The API can process unit readings in various formats:

- `Unit 1A reads 123.45 cubic meter`
- `Unit 2B is 67 m3`
- `3C reading 101.1 m³`
- `Unit 4D: 202.20 cubic meter`
- `5E = 303 m3`
- `Unit 6F reads 404.4 cu.m`

## 🔧 Configuration

### Environment Variables

- `GOOGLE_API_KEY`: Your Google Gemini API key (required)
- `FLASK_ENV`: Set to `development` for debug mode
- `PORT`: Port number for the Flask app (default: 8080)



## 📊 Processing Flow

1. **Input Reception**: API receives text input
2. **Gemini Standardization**: AI processes and standardizes the input
3. **Regex Extraction**: Pattern matching extracts unit-reading pairs
4. **Validation**: Checks for duplicates and conflicts
5. **Data Cleaning**: Removes duplicates and resolves conflicts
6. **JSON Output**: Returns structured data

