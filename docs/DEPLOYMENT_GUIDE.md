# Deployment Guide

## Overview

This guide covers deploying the SocialMedia Sentiment Analysis system in various environments, from development to production.

## Deployment Options

### 1. Local Development Deployment

#### Quick Setup
```bash
# Clone and setup
git clone <repository-url>
cd SocialMedia_SentimentAnalysis
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
python -c "import nltk; nltk.download('all')"

# Test deployment
python run_pipeline.py --quick-test
```

#### Development Server
```bash
# For interactive development
jupyter notebook main.ipynb

# For automated testing
pytest tests/ -v
```

### 2. Production Server Deployment

#### System Requirements
- **OS**: Ubuntu 20.04+ / CentOS 8+ / Amazon Linux 2
- **CPU**: 4+ cores (8+ recommended for large datasets)
- **RAM**: 8GB minimum (16GB+ recommended)
- **Storage**: 10GB free space
- **Python**: 3.12+

#### Installation Steps

##### Step 1: System Preparation
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python and dependencies
sudo apt install -y python3.12 python3.12-venv python3.12-dev
sudo apt install -y build-essential gcc g++

# Create application user
sudo useradd -m -s /bin/bash sentiment_app
sudo usermod -aG sudo sentiment_app
```

##### Step 2: Application Setup
```bash
# Switch to app user
sudo su - sentiment_app

# Create application directory
sudo mkdir -p /opt/sentiment_analysis
sudo chown sentiment_app:sentiment_app /opt/sentiment_analysis
cd /opt/sentiment_analysis

# Clone repository
git clone <repository-url> .

# Setup virtual environment
python3.12 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('all')"
```

##### Step 3: Configuration
```bash
# Create configuration file
cat > config/production.conf << EOF
[DEFAULT]
data_dir = /data/sentiment
model_dir = /models/sentiment
log_dir = /var/log/sentiment
max_features = 10000
n_jobs = -1
log_level = INFO
EOF

# Create directories
sudo mkdir -p /data/sentiment /models/sentiment /var/log/sentiment
sudo chown -R sentiment_app:sentiment_app /data/sentiment /models/sentiment /var/log/sentiment
```

##### Step 4: Service Setup
```bash
# Create systemd service
sudo tee /etc/systemd/system/sentiment-analysis.service << EOF
[Unit]
Description=Sentiment Analysis Pipeline
After=network.target

[Service]
Type=simple
User=sentiment_app
Group=sentiment_app
WorkingDirectory=/opt/sentiment_analysis
Environment=PATH=/opt/sentiment_analysis/venv/bin
Environment=PYTHONPATH=/opt/sentiment_analysis
ExecStart=/opt/sentiment_analysis/venv/bin/python run_pipeline.py
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

# Enable and start service
sudo systemctl daemon-reload
sudo systemctl enable sentiment-analysis
sudo systemctl start sentiment-analysis

# Check status
sudo systemctl status sentiment-analysis
```

### 3. Docker Deployment

#### Dockerfile
```dockerfile
FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create app user
RUN useradd -m -s /bin/bash sentiment_app

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data
RUN python -c "import nltk; nltk.download('all')"

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p datasets features models reports/eda reports/evaluation

# Change ownership
RUN chown -R sentiment_app:sentiment_app /app

# Switch to app user
USER sentiment_app

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import src; print('OK')" || exit 1

# Default command
CMD ["python", "run_pipeline.py"]
```

#### Docker Compose
```yaml
version: '3.8'

services:
  sentiment-analysis:
    build: .
    container_name: sentiment_app
    volumes:
      - ./data:/app/datasets:ro
      - sentiment_models:/app/models
      - sentiment_reports:/app/reports
    environment:
      - MAX_FEATURES=10000
      - N_JOBS=4
      - LOG_LEVEL=INFO
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "python", "-c", "import src; print('OK')"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s

  # Optional: Monitoring
  prometheus:
    image: prom/prometheus:latest
    container_name: prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml:ro
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'

volumes:
  sentiment_models:
  sentiment_reports:
```

#### Deployment Commands
```bash
# Build and deploy
docker-compose build
docker-compose up -d

# View logs
docker-compose logs -f sentiment-analysis

# Scale service
docker-compose up -d --scale sentiment-analysis=3

# Update deployment
docker-compose pull
docker-compose up -d --force-recreate
```

### 4. Kubernetes Deployment

#### Namespace and ConfigMap
```yaml
# namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: sentiment-analysis

---
# configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: sentiment-config
  namespace: sentiment-analysis
data:
  MAX_FEATURES: "10000"
  N_JOBS: "4"
  LOG_LEVEL: "INFO"
```

#### Deployment
```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: sentiment-analysis
  namespace: sentiment-analysis
  labels:
    app: sentiment-analysis
spec:
  replicas: 3
  selector:
    matchLabels:
      app: sentiment-analysis
  template:
    metadata:
      labels:
        app: sentiment-analysis
    spec:
      containers:
      - name: sentiment-app
        image: sentiment-analysis:latest
        ports:
        - containerPort: 8000
        envFrom:
        - configMapRef:
            name: sentiment-config
        resources:
          requests:
            memory: "2Gi"
            cpu: "1"
          limits:
            memory: "4Gi"
            cpu: "2"
        volumeMounts:
        - name: data-volume
          mountPath: /app/datasets
          readOnly: true
        - name: models-volume
          mountPath: /app/models
        - name: reports-volume
          mountPath: /app/reports
        livenessProbe:
          exec:
            command:
            - python
            - -c
            - "import src; print('OK')"
          initialDelaySeconds: 60
          periodSeconds: 30
        readinessProbe:
          exec:
            command:
            - python
            - -c
            - "import src; print('OK')"
          initialDelaySeconds: 30
          periodSeconds: 10
      volumes:
      - name: data-volume
        persistentVolumeClaim:
          claimName: sentiment-data-pvc
      - name: models-volume
        persistentVolumeClaim:
          claimName: sentiment-models-pvc
      - name: reports-volume
        persistentVolumeClaim:
          claimName: sentiment-reports-pvc
```

#### Service and Ingress
```yaml
# service.yaml
apiVersion: v1
kind: Service
metadata:
  name: sentiment-service
  namespace: sentiment-analysis
spec:
  selector:
    app: sentiment-analysis
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: ClusterIP

---
# ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: sentiment-ingress
  namespace: sentiment-analysis
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
spec:
  rules:
  - host: sentiment.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: sentiment-service
            port:
              number: 80
```

#### Deploy to Kubernetes
```bash
# Apply configurations
kubectl apply -f namespace.yaml
kubectl apply -f configmap.yaml
kubectl apply -f pvc.yaml
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml
kubectl apply -f ingress.yaml

# Check deployment
kubectl get pods -n sentiment-analysis
kubectl logs -f deployment/sentiment-analysis -n sentiment-analysis

# Scale deployment
kubectl scale deployment sentiment-analysis --replicas=5 -n sentiment-analysis
```

## Environment Configuration

### Environment Variables

#### Production Environment
```bash
# Application settings
export SENTIMENT_ENV=production
export SENTIMENT_DATA_DIR=/data/sentiment
export SENTIMENT_MODEL_DIR=/models/sentiment
export SENTIMENT_LOG_DIR=/var/log/sentiment

# Performance settings
export MAX_FEATURES=10000
export N_JOBS=-1
export BATCH_SIZE=5000

# Logging
export LOG_LEVEL=INFO
export LOG_FORMAT=json

# Security
export SECURE_MODE=true
export API_KEY_REQUIRED=true
```

#### Development Environment
```bash
export SENTIMENT_ENV=development
export MAX_FEATURES=2000
export N_JOBS=2
export LOG_LEVEL=DEBUG
export QUICK_TEST=true
```

### Configuration Files

#### Production Config (`config/production.ini`)
```ini
[application]
environment = production
debug = false
max_features = 10000
n_jobs = -1

[logging]
level = INFO
format = json
file = /var/log/sentiment/app.log
max_size = 100MB
backup_count = 5

[data]
input_dir = /data/sentiment/input
output_dir = /data/sentiment/output
model_dir = /models/sentiment

[performance]
batch_size = 5000
memory_limit = 8GB
timeout = 3600
```

## Monitoring and Logging

### Application Monitoring

#### Health Check Endpoint
```python
# health.py
from datetime import datetime
import psutil
import os

def health_check():
    """Application health check"""
    return {
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0',
        'uptime': get_uptime(),
        'memory_usage': psutil.virtual_memory().percent,
        'disk_usage': psutil.disk_usage('/').percent,
        'cpu_usage': psutil.cpu_percent(interval=1)
    }

def get_uptime():
    """Get application uptime"""
    with open('/proc/uptime', 'r') as f:
        uptime_seconds = float(f.readline().split()[0])
    return uptime_seconds
```

#### Logging Configuration
```python
# logging_config.py
import logging
import logging.handlers
import json
from datetime import datetime

class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
            
        return json.dumps(log_entry)

def setup_logging():
    """Setup production logging"""
    logger = logging.getLogger('sentiment_analysis')
    logger.setLevel(logging.INFO)
    
    # File handler with rotation
    file_handler = logging.handlers.RotatingFileHandler(
        '/var/log/sentiment/app.log',
        maxBytes=100*1024*1024,  # 100MB
        backupCount=5
    )
    file_handler.setFormatter(JSONFormatter())
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(JSONFormatter())
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger
```

### System Monitoring

#### Prometheus Metrics
```python
# metrics.py
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import time
import psutil

# Metrics
pipeline_runs_total = Counter('pipeline_runs_total', 'Total pipeline runs')
pipeline_duration = Histogram('pipeline_duration_seconds', 'Pipeline execution time')
model_accuracy = Gauge('model_accuracy', 'Current model accuracy')
memory_usage = Gauge('memory_usage_percent', 'Memory usage percentage')

def track_pipeline_run(func):
    """Decorator to track pipeline metrics"""
    def wrapper(*args, **kwargs):
        pipeline_runs_total.inc()
        
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            pipeline_duration.observe(time.time() - start_time)
            
            # Update accuracy if available
            if 'accuracy' in result:
                model_accuracy.set(result['accuracy'])
                
            return result
        except Exception as e:
            pipeline_duration.observe(time.time() - start_time)
            raise
    return wrapper

def update_system_metrics():
    """Update system metrics"""
    memory_usage.set(psutil.virtual_memory().percent)

# Start metrics server
start_http_server(8000)
```

## Backup and Recovery

### Automated Backup
```bash
#!/bin/bash
# backup.sh

BACKUP_DIR="/backup/sentiment_analysis"
DATE=$(date +%Y%m%d_%H%M%S)
RETENTION_DAYS=7

# Create backup directory
mkdir -p "$BACKUP_DIR/$DATE"

# Backup models
echo "Backing up models..."
tar -czf "$BACKUP_DIR/$DATE/models.tar.gz" /models/sentiment/

# Backup processed data
echo "Backing up data..."
tar -czf "$BACKUP_DIR/$DATE/data.tar.gz" /data/sentiment/

# Backup configuration
echo "Backing up configuration..."
cp -r /opt/sentiment_analysis/config/ "$BACKUP_DIR/$DATE/"

# Backup logs (last 7 days)
echo "Backing up logs..."
find /var/log/sentiment/ -name "*.log*" -mtime -7 -exec cp {} "$BACKUP_DIR/$DATE/" \;

# Cleanup old backups
echo "Cleaning up old backups..."
find "$BACKUP_DIR" -type d -mtime +$RETENTION_DAYS -exec rm -rf {} \;

echo "Backup completed: $BACKUP_DIR/$DATE"

# Upload to cloud storage (optional)
# aws s3 sync "$BACKUP_DIR/$DATE" s3://sentiment-backups/$DATE/
```

### Recovery Procedures
```bash
#!/bin/bash
# restore.sh

BACKUP_DATE=$1
BACKUP_DIR="/backup/sentiment_analysis/$BACKUP_DATE"

if [ -z "$BACKUP_DATE" ]; then
    echo "Usage: $0 <backup_date>"
    echo "Available backups:"
    ls -1 /backup/sentiment_analysis/
    exit 1
fi

if [ ! -d "$BACKUP_DIR" ]; then
    echo "Backup not found: $BACKUP_DIR"
    exit 1
fi

echo "Stopping services..."
sudo systemctl stop sentiment-analysis

echo "Restoring models..."
tar -xzf "$BACKUP_DIR/models.tar.gz" -C /

echo "Restoring data..."
tar -xzf "$BACKUP_DIR/data.tar.gz" -C /

echo "Restoring configuration..."
cp -r "$BACKUP_DIR/config/" /opt/sentiment_analysis/

echo "Starting services..."
sudo systemctl start sentiment-analysis

echo "Recovery completed from backup: $BACKUP_DATE"
```

## Security Considerations

### Access Control
```bash
# Set proper file permissions
chmod 750 /opt/sentiment_analysis
chmod 640 /opt/sentiment_analysis/config/*
chmod 600 /opt/sentiment_analysis/config/secrets.conf

# Restrict log access
chmod 750 /var/log/sentiment
chmod 640 /var/log/sentiment/*.log
```

### Network Security
```bash
# Firewall rules (UFW)
sudo ufw allow ssh
sudo ufw allow 8000/tcp  # Application port
sudo ufw enable

# Or iptables
sudo iptables -A INPUT -p tcp --dport 8000 -j ACCEPT
sudo iptables -A INPUT -p tcp --dport 22 -j ACCEPT
sudo iptables -P INPUT DROP
```

### Data Protection
- Encrypt sensitive data at rest
- Use secure communication channels
- Implement proper authentication
- Regular security updates
- Monitor access logs

This deployment guide provides comprehensive instructions for deploying the sentiment analysis system across different environments with proper monitoring, backup, and security considerations.
