# Remorph - Advanced Deepfake Detection System

A comprehensive deepfake detection and attribution system with CPU-first forensic analysis, self-learning capabilities, and robust API infrastructure.

## Features

### Core Detection
- **Deep Learning Models**: Support for both Hugging Face transformers and local PyTorch models
- **Forensic Feature Extraction**: Comprehensive multi-modal feature analysis including:
  - Residual noise analysis
  - Spectral domain features (FFT/DCT)
  - Color statistics and cross-channel correlations
  - Texture analysis (LBP, GLCM, Gabor filters)
  - Compression artifact detection
  - CNN embeddings
- **Face Detection**: MTCNN-based face detection with confidence scoring
- **Attribution Engine**: Ensemble attribution with closed-set, open-set, and embedding methods

### Advanced Analysis
- **Robustness Testing**: Comprehensive manipulation resistance testing
- **Self-Learning System**: Active learning with user consent management
- **Quality Assessment**: Multi-factor quality scoring for training suitability
- **Forensic Reports**: Automated generation of technical and legal reports

### API & Infrastructure
- **FastAPI Backend**: Modern async API with comprehensive endpoints
- **Job Processing**: Background job processing with Redis support
- **Batch Analysis**: Concurrent processing of multiple images
- **Real-time Updates**: Server-Sent Events for live progress tracking
- **Rate Limiting**: Built-in protection against abuse

### Frontend
- **Modern React UI**: Responsive interface built with React 18 and TypeScript
- **Real-time Monitoring**: Live system health and statistics
- **Interactive Visualizations**: Heatmaps, overlays, and forensic analysis results
- **Batch Processing**: Multi-image upload and analysis
- **Learning Management**: Consent workflow and candidate management

## Quick Start

### Prerequisites
- Python 3.11+
- Node.js 18+
- Redis (optional, for production job processing)
- Docker (optional, for Redis)

### Backend Setup

1. **Create virtual environment**:
   ```bash
   cd backend
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment** (optional):
   ```bash
   cp .env.example .env
   # Edit .env with your preferences
   ```

4. **Start the API server**:
   ```bash
   python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8080 --reload
   ```

### Frontend Setup

1. **Install dependencies**:
   ```bash
   cd frontend
   npm install
   ```

2. **Start development server**:
   ```bash
   npm run dev
   ```

The frontend will be available at `http://localhost:3000` and will proxy API requests to the backend at `http://localhost:8080`.

### Using Docker (Redis)

For production-grade job processing with Redis:

```bash
# Start Redis
docker compose up -d

# Set Redis URL and start backend
export REDIS_URL=redis://localhost:6379/0
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8080
```

## API Documentation

Once the backend is running, visit `http://localhost:8080/docs` for interactive API documentation.

### Key Endpoints

- `POST /analyze` - Analyze single image (synchronous)
- `POST /analyze/submit` - Submit analysis job (asynchronous)
- `GET /analyze/events/{job_id}` - Server-Sent Events for job progress
- `POST /analyze/batch` - Batch analysis of multiple images
- `GET /health` - System health check
- `GET /stats` - System statistics
- `GET /families` - Attribution families

## Configuration

### Model Configuration

The system supports two model providers:

1. **Hugging Face Models** (recommended):
   ```bash
   export MODEL_PROVIDER=huggingface
   export MODEL_NAME=Hemg/Deepfake-Detection
   ```

2. **Local PyTorch Models**:
   ```bash
   export MODEL_PROVIDER=local_torch
   export WEIGHTS_PATH=weights/detector.pt
   ```

### Environment Variables

Key configuration options (see `backend/.env.example`):

- `OUTPUT_DIR`: Directory for generated files (default: `outputs`)
- `MAX_FILE_SIZE_MB`: Maximum upload size (default: `10`)
- `FACE_CONFIDENCE_THRESHOLD`: Minimum face detection confidence (default: `0.90`)
- `REDIS_URL`: Redis connection for job processing (optional)

## Development

### Running Tests

```bash
# Backend tests
cd backend
pytest

# With Redis tests (requires Redis running)
export REDIS_URL=redis://localhost:6379/0
pytest tests/test_redis_job_store.py

# Feature extraction tests
python scripts/run_feature_tests.py
```

### Code Quality

```bash
# Type checking
mypy backend/src

# Linting (if configured)
flake8 backend/src
```

### Maintenance

```bash
# Cleanup old outputs
python backend/scripts/maintenance.py cleanup --days 7

# Backup fingerprints
python backend/scripts/maintenance.py backup

# Validate fingerprints integrity
python backend/scripts/maintenance.py validate
```

## Architecture

### Backend Structure
```
backend/
├── src/
│   ├── api/           # FastAPI routes and middleware
│   ├── core/          # Core analysis engines
│   ├── models/        # Detection models and face detection
│   ├── utils/         # Utilities and helpers
│   ├── trace/         # Attribution and fingerprinting
│   └── ingest/        # Image processing and filtering
├── tests/             # Test suite
├── scripts/           # Maintenance and utility scripts
└── data/              # Fingerprint database
```

### Frontend Structure
```
frontend/
├── src/
│   ├── components/    # React components
│   ├── services/      # API client and utilities
│   └── types/         # TypeScript type definitions
├── public/            # Static assets
└── dist/              # Built application
```

## Migration Notes

This version includes several breaking changes from earlier versions:

1. **API Response Format**: `legacy_deepfake_score` has been removed. Use `deep_model_score` instead.
2. **Model Configuration**: New unified model provider system supports both HF and local models.
3. **Job Processing**: Enhanced background job system with SSE support.
4. **Self-Learning**: Complete rewrite with consent management and safeguards.

See `MIGRATION.md` and `DEPRECATION.md` for detailed migration guidance.

## Production Deployment

### Backend
1. Set production environment variables
2. Use a production WSGI server (gunicorn recommended)
3. Configure Redis for job processing
4. Set up proper logging and monitoring
5. Configure reverse proxy (nginx recommended)

### Frontend
1. Build the production bundle: `npm run build`
2. Serve static files with a web server
3. Configure API proxy to backend
4. Set up SSL/TLS certificates

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes with tests
4. Ensure all tests pass
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For issues, questions, or contributions:
- Open an issue on GitHub
- Check the documentation in `/docs`
- Review the API documentation at `/docs` when running

## Acknowledgments

- Built with FastAPI, React, and modern ML frameworks
- Uses MTCNN for face detection
- Supports Hugging Face transformers ecosystem
- Inspired by forensic analysis techniques and active learning research