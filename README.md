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

## Local Development (Recommended for Apple Silicon)

For faster iteration during development, use a local Python virtual environment.

### 1. Setup Virtual Environment

```bash
# Create virtual environment with Python 3.9
python3.9 -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2.5. Set Up Environment Variables

Create a `.env` file in the project root with your API credentials:

```bash
cp .env.example .env
```

Then edit `.env` and add your credentials:
```
ROBOFLOW_API_KEY=your_roboflow_api_key_here
```

Get your Roboflow API key from: https://app.roboflow.com/account

**Note:** The `.env` file is git-ignored for security. Never commit it to version control.

### 3. Run Your Scripts

Once activated, run scripts directly:
```bash
# With venv activated
python path/to/script.py

# Or without activating (using explicit interpreter)
./venv/bin/python path/to/script.py
```

### 4. Deactivate Virtual Environment

```bash
deactivate
```

### 5. Long-term Usage

```bash
# Each time you open a new terminal in this project
source venv/bin/activate

# Then run your code
python your_script.py

# When done
deactivate
```

### Verify Installation

```bash
./venv/bin/python --version  # Should show Python 3.9.23
./venv/bin/python -c "import torch, cv2, transformers; print('All packages available')"
```

### Note on GPU Support

The current setup uses CPU-based PyTorch. For Metal GPU acceleration on Apple Silicon:
```bash
pip install --upgrade torch torchvision
```

This will install Metal-accelerated versions if available for your Python version.

## Model Training (Cloud)

- Train models on cloud infrastructure (GPU instances)
- Download trained models and place in `models/` directory
- Use Docker or local venv for inference/tracking

## Next Steps

1. Configure your main script entry point
2. Add data directories (they're git-ignored for safety)
3. Start developing within Docker containers
4. Set up cloud training pipeline separately
