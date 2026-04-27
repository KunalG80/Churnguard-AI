FROM python:3.11-slim

WORKDIR /app

# System deps for reportlab + matplotlib
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libfreetype6-dev \
    fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Pre-create output dirs
RUN mkdir -p data/raw data/processed models reports/figures assets/fonts

# Copy DejaVu font from system
RUN cp /usr/share/fonts/truetype/dejavu/DejaVuSans.ttf assets/fonts/ 2>/dev/null || true

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Default: launch dashboard
# To train first: docker exec <container> python main.py
CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
