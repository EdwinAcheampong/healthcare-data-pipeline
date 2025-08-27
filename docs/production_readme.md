# Phase 3: Production API & Deployment

## 🎯 Overview

Phase 3 of the Healthcare Data Pipeline project focuses on production deployment of the API, comprehensive monitoring, and real-time healthcare workload optimization. This phase brings together the ML models from Phase 2A and RL optimization from Phase 2B into a production-ready API system.

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │   API Gateway   │    │   ML/RL Engine  │
│   Dashboard     │◄──►│   (FastAPI)     │◄──►│   (Phase 2A/B)  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Monitoring    │    │   Database      │    │   Cache Layer   │
│   (Prometheus)  │    │   (PostgreSQL)  │    │   (Redis)       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🚀 Quick Start

### 1. Prerequisites

- Docker and Docker Compose
- Python 3.11+
- 8GB+ RAM (for full monitoring stack)

### 2. Environment Setup

```bash
# Clone the repository
git clone <repository-url>
cd healthcare-data-pipeline

# Copy environment file
cp env.example .env

# Edit environment variables
nano .env
```

### 3. Start Development Environment

```bash
# Start all services
docker-compose up -d

# Check services
docker-compose ps

# View logs
docker-compose logs -f api
```

### 4. Test the API

```bash
# Run API tests
python scripts/test_phase3_api.py

# Or test manually
curl http://localhost:8000/health
```

## 📚 API Documentation

### Base URL

- **Development**: `http://localhost:8000`
- **Production**: `https://your-domain.com`

### Interactive Documentation

- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

## 🔧 API Endpoints

### Health & Status

- `GET /` - API information
- `GET /health` - Health check
- `GET /api/v1/health` - Detailed health status
- `GET /api/v1/status` - System status

### Optimization Endpoints

- `POST /api/v1/optimize/workload` - Optimize healthcare workload
- `GET /api/v1/optimize/status/{job_id}` - Get optimization status
- `GET /api/v1/optimize/history` - Optimization history
- `GET /api/v1/optimize/strategies` - Available strategies

### Prediction Endpoints

- `POST /api/v1/predict/workload` - Predict workload
- `POST /api/v1/predict/batch` - Batch predictions
- `GET /api/v1/predict/models` - Available models
- `GET /api/v1/predict/performance` - Model performance

### Monitoring Endpoints

- `GET /api/v1/monitoring/metrics` - System metrics
- `GET /api/v1/monitoring/dashboard` - Dashboard data
- `GET /api/v1/monitoring/alerts` - System alerts

## 📊 Example Usage

### 1. Workload Optimization

```python
import requests

# Optimize healthcare workload
optimization_request = {
    "current_patients": 50,
    "current_staff": 20,
    "department": "emergency",
    "shift_hours": 8,
    "optimization_strategy": "ppo",
    "constraints": {"max_staff": 25},
    "target_metrics": ["efficiency", "patient_satisfaction"]
}

response = requests.post(
    "http://localhost:8000/api/v1/optimize/workload",
    json=optimization_request
)

result = response.json()
print(f"Optimization ID: {result['optimization_id']}")
print(f"Recommended Staff: {result['recommended_staff']}")
print(f"Efficiency Gain: {result['efficiency_gain']:.2%}")
```

### 2. Workload Prediction

```python
# Predict workload
prediction_request = {
    "model_type": "advanced",
    "input_data": {
        "patient_count": 45,
        "time_of_day": 14,
        "day_of_week": 2,
        "department": "emergency"
    },
    "prediction_horizon": 24,
    "confidence_level": 0.95
}

response = requests.post(
    "http://localhost:8000/api/v1/predict/workload",
    json=prediction_request
)

result = response.json()
print(f"Predicted Value: {result['predicted_value']}")
print(f"Confidence: {result['model_confidence']:.2%}")
```

### 3. System Monitoring

```python
# Get system metrics
response = requests.get("http://localhost:8000/api/v1/monitoring/metrics")
metrics = response.json()

print(f"CPU Usage: {metrics['system_metrics']['cpu_usage']:.1f}%")
print(f"Memory Usage: {metrics['system_metrics']['memory_usage']:.1f}%")
print(f"API Response Time: {metrics['api_metrics']['average_response_time']:.3f}s")
```

## 🐳 Production Deployment

### 1. Production Environment

```bash
# Start production stack
docker-compose -f docker-compose.prod.yml up -d

# Check all services
docker-compose -f docker-compose.prod.yml ps
```

### 2. Environment Variables

Create `.env.prod` with production values:

```bash
# API Configuration
ENVIRONMENT=production
JWT_SECRET_KEY=your-secure-jwt-secret
CORS_ORIGINS=https://yourdomain.com

# Database
DATABASE_URL=postgresql://user:pass@db:5432/healthcare_prod
REDIS_URL=redis://redis:6379

# Monitoring
GRAFANA_PASSWORD=secure-grafana-password
```

### 3. Health Checks

```bash
# Check API health
curl http://localhost:8000/health

# Check all services
curl http://localhost:8000/api/v1/health

# Check monitoring
curl http://localhost:9090/-/healthy  # Prometheus
curl http://localhost:3001/api/health # Grafana
```

## 📈 Monitoring & Observability

### Available Dashboards

1. **Grafana** (`http://localhost:3001`)

   - System metrics dashboard
   - API performance dashboard
   - Model performance dashboard
   - Business KPIs dashboard

2. **Prometheus** (`http://localhost:9090`)

   - Metrics collection and storage
   - Alert rules and configuration

3. **Jaeger** (`http://localhost:16686`)

   - Distributed tracing
   - Request flow analysis

4. **Kibana** (`http://localhost:5601`)
   - Log aggregation and analysis
   - Error tracking and debugging

### Key Metrics

- **API Response Time**: Target < 200ms
- **Error Rate**: Target < 1%
- **System Uptime**: Target 99.9%
- **Model Performance**: MAPE < 8%
- **Compliance Rate**: Target > 95%

## 🔒 Security

### Authentication

- JWT-based authentication
- Role-based access control
- API key management

### Security Headers

- CORS configuration
- Trusted host middleware
- Rate limiting
- Input validation

### Data Protection

- HTTPS encryption
- Database connection encryption
- Audit logging
- PHI anonymization

## 🧪 Testing

### Run API Tests

```bash
# Run comprehensive API tests
python scripts/test_phase3_api.py

# Test specific endpoint
python scripts/test_phase3_api.py http://localhost:8000
```

### Load Testing

```bash
# Install locust
pip install locust

# Run load test
locust -f tests/load/locustfile.py --host=http://localhost:8000
```

### Integration Tests

```bash
# Run integration tests
pytest tests/integration/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## 🔧 Configuration

### API Configuration

```python
# src/config/settings.py
class APISettings(BaseSettings):
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: list = ["http://localhost:3000"]
    allowed_hosts: list = ["localhost", "127.0.0.1"]
    log_level: str = "INFO"
```

### Database Configuration

```python
class DatabaseSettings(BaseSettings):
    database_url: str = "postgresql://user:pass@localhost:5432/healthcare"
    redis_url: str = "redis://localhost:6379"
```

## 📊 Performance Optimization

### Caching Strategy

- Redis caching for predictions
- Model result caching
- API response caching

### Database Optimization

- Connection pooling
- Query optimization
- Index management

### Model Optimization

- Model quantization
- Batch processing
- Async prediction

## 🚨 Troubleshooting

### Common Issues

1. **API Not Starting**

   ```bash
   # Check logs
   docker-compose logs api

   # Check dependencies
   docker-compose ps
   ```

2. **Database Connection Issues**

   ```bash
   # Test database connection
   docker-compose exec db psql -U healthcare_user -d healthcare_prod

   # Check database logs
   docker-compose logs db
   ```

3. **Model Loading Issues**

   ```bash
   # Check model files
   ls -la models/

   # Test model import
   python -c "from src.models.baseline_models import BaselinePredictor"
   ```

### Log Analysis

```bash
# View API logs
docker-compose logs -f api

# View all logs
docker-compose logs -f

# Search for errors
docker-compose logs | grep ERROR
```

## 📚 Additional Resources

### Documentation

- [API Reference](http://localhost:8000/docs)
- [Deployment Guide](docs/PHASE_3_DEPLOYMENT_GUIDE.md)
- [Architecture Overview](docs/PROJECT_STRUCTURE.md)

### Monitoring

- [Grafana Dashboards](http://localhost:3001)
- [Prometheus Metrics](http://localhost:9090)
- [Jaeger Traces](http://localhost:16686)

### Support

- [Issue Tracker](https://github.com/your-repo/issues)
- [Documentation](https://github.com/your-repo/docs)
- [Community Forum](https://github.com/your-repo/discussions)

## 🎉 Success Metrics

Phase 3 is considered successful when:

- ✅ API response time < 200ms
- ✅ 99.9% uptime achieved
- ✅ < 1% error rate
- ✅ Real-time optimization working
- ✅ Comprehensive monitoring active
- ✅ Security requirements met
- ✅ Documentation complete

---

**Phase 3 Status**: ✅ **COMPLETE**

The healthcare data pipeline is now production-ready with a comprehensive API, monitoring stack, and deployment infrastructure.
