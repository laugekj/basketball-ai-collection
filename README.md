# Basketball AI Collection

ML project for basketball tracking, detection, and analysis.

## Project Structure

```
basketball-ai-collection/
├── common/              # Common utilities and classes
├── models/              # Model weights (git-ignored)
├── input_videos/        # Input video files (git-ignored)
├── output_videos/       # Output processed videos (git-ignored)
├── results/             # Analysis results and outputs (git-ignored)
├── logs/                # Logs (git-ignored)
├── Dockerfile           # Docker container definition
├── docker-compose.yml   # Docker compose configuration
├── requirements.txt     # Python dependencies
└── README.md            # This file
```

## Development Setup with Docker

### 1. Build the Docker Image

```bash
docker-compose build
```

### 2. Run Interactive Development Session

```bash
docker-compose run --rm basketball-ai /bin/bash
```

Then inside the container:
```bash
python -m basketball_ai.main
```

### 3. Run a Specific Script

```bash
docker-compose run --rm basketball-ai python path/to/script.py
```

### 4. Run Tests

```bash
docker-compose run --rm basketball-ai pytest
```

## Important Notes on Apple Silicon

⚠️ **GPU Support**: Docker on Apple Silicon doesn't provide direct access to Metal GPU. Options:

1. **CPU-based development** (recommended for Apple Silicon):
   - Develop locally with a venv (faster iteration)
   - Use Docker for consistency testing only
   - Train on cloud with full GPU support

2. **Docker CPU mode** (current setup):
   - Works everywhere, but slower for inference
   - Good for testing code before cloud deployment

## Alternative: Local Virtual Environment

If you prefer faster iteration on your Mac with GPU (Metal):

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run scripts directly
python main.py
```

For GPU support with local venv on Apple Silicon:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cpu
```

## Model Training (Cloud)

- Train models on cloud infrastructure (GPU instances)
- Download trained models and place in `models/` directory
- Use Docker or local venv for inference/tracking

## Next Steps

1. Configure your main script entry point
2. Add data directories (they're git-ignored for safety)
3. Start developing within Docker containers
4. Set up cloud training pipeline separately
