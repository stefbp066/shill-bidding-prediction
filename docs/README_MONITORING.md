# Model Monitoring Setup

This project includes a comprehensive monitoring setup using Evidently AI, Prometheus, and Grafana for the Shill Bidding Prediction model.

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   FastAPI App   │───▶│   Prometheus    │───▶│    Grafana      │
│   (Port 8000)   │    │   (Port 9090)   │    │   (Port 3000)   │
│ + Evidently AI  │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │                       │
                                ▼                       ▼
                       ┌─────────────────┐    ┌─────────────────┐
                       │  Evidently      │    │  Model Monitoring│
                       │   Cloud         │    │   Dashboards    │
                       └─────────────────┘    └─────────────────┘
```

## Features

### 🎯 Model Monitoring
- **Data Drift Detection**: Monitors feature distribution changes using Evidently AI presets
- **Model Performance**: Tracks classification metrics over time
- **Prediction Monitoring**: Real-time monitoring of predictions and probabilities

### 📊 Metrics & Visualization
- **Prometheus Metrics**: Custom metrics for prediction count, latency, drift scores
- **Grafana Dashboards**: Pre-configured dashboards for model monitoring
- **Real-time Alerts**: Configurable thresholds for drift and performance

### 🔧 Monitoring Presets Used
- **DataDriftPreset**: Detects data drift in features
- **ClassificationPreset**: Monitors classification performance metrics
- **DataSummaryPreset**: Provides data quality insights

## Quick Start

### 1. Start the Monitoring Stack

```bash
# Start all services
docker-compose up -d

# Check service status
docker-compose ps
```

### 2. Access the Services

- **API**: http://localhost:8000
- **Grafana**: http://localhost:3000 (admin/admin)
- **Prometheus**: http://localhost:9090

### 3. Test the Monitoring

```bash
# Run the test script
python test_monitoring.py
```

## Configuration

### Environment Variables

Create a `.env` file in the root directory:

```env
# Evidently Cloud (optional)
EVIDENTLY_TOKEN=your_evidently_cloud_token

# API Configuration
API_PORT=8000
MLFLOW_TRACKING_URI=mlruns
MODEL_NAME=shill-bidding-model
MODEL_VERSION=latest
```

### Prometheus Configuration

The Prometheus configuration is in `monitoring/prometheus.yml`:

```yaml
scrape_configs:
  - job_name: 'api'
    static_configs:
      - targets: ['api:8000']
    metrics_path: '/metrics'
    scrape_interval: 10s
```

### Grafana Dashboards

Pre-configured dashboards are located in `monitoring/grafana/dashboards/`:

- **Model Monitoring Dashboard**: Comprehensive view of model performance
- **Data Drift Dashboard**: Feature drift analysis
- **Prediction Analytics**: Real-time prediction metrics

## API Endpoints

### Prediction Endpoints

```bash
# Single prediction with monitoring
POST /predict
{
  "auction_id": "auction_001",
  "bidder_tendency": 0.5,
  "bidding_ratio": 0.3,
  "successive_outbidding": 0.2,
  "last_bidding": 0.8,
  "auction_bids": 0.4,
  "starting_price_average": 0.6,
  "early_bidding": 0.1,
  "winning_ratio": 0.7
}

# Response includes monitoring data
{
  "prediction": 1,
  "probability": 0.85,
  "is_shill_bid": true,
  "drift_score": 0.23,
  "monitoring_info": {...}
}
```

### Monitoring Endpoints

```bash
# Prometheus metrics
GET /metrics

# Health check
GET /health

# Model information
GET /model-info
```

## Monitoring Metrics

### Prometheus Metrics

- `evidently_prediction_count_total`: Total number of predictions
- `evidently_prediction_errors_total`: Total number of errors
- `evidently_prediction_duration_seconds`: Prediction latency
- `evidently_data_drift_score`: Data drift score (0-1)
- `evidently_model_performance_score`: Model performance score

### Evidently AI Reports

The monitoring system generates three types of reports:

1. **Data Drift Reports**: Compare current data with reference data
2. **Classification Reports**: Monitor model performance metrics
3. **Data Summary Reports**: Data quality and distribution analysis

## Dashboard Features

### Model Monitoring Dashboard

- **Data Drift Score**: Real-time drift detection with color-coded thresholds
- **Model Performance**: Accuracy, precision, recall over time
- **Prediction Rate**: Requests per second
- **Error Rate**: Failed predictions per second
- **Latency**: 95th percentile response time

### Alerting

Configure alerts in Grafana for:

- **High Drift Score**: > 0.7 (red threshold)
- **Low Performance**: < 0.7 (yellow threshold)
- **High Error Rate**: > 5% of predictions
- **High Latency**: > 2 seconds

## Development

### Adding New Metrics

1. Add Prometheus metrics in `api/main.py`:
```python
NEW_METRIC = Gauge('new_metric_name', 'Description')
```

2. Update the monitoring in prediction functions:
```python
NEW_METRIC.set(value)
```

### Custom Evidently Reports

1. Create new report types in `monitoring/monitoring_evidently.py`:
```python
def create_custom_report(self, current_data: pd.DataFrame) -> Report:
    report = Report(metrics=[CustomMetric()])
    report.run(reference_data=self.reference_dataset.data, current_data=current_data)
    return report
```

2. Add to the monitoring pipeline in the API.

## Troubleshooting

### Common Issues

1. **Evidently Cloud Connection Failed**
   - Check `EVIDENTLY_TOKEN` environment variable
   - Verify network connectivity
   - Monitor will continue without cloud logging

2. **Prometheus Not Scraping**
   - Check service names in `docker-compose.yml`
   - Verify metrics endpoint at `/metrics`
   - Check Prometheus targets at http://localhost:9090/targets

3. **Grafana Dashboard Not Loading**
   - Verify Prometheus datasource configuration
   - Check dashboard JSON format
   - Restart Grafana container

### Logs

```bash
# View all logs
docker-compose logs

# View specific service logs
docker-compose logs api
docker-compose logs prometheus
docker-compose logs grafana
```

## Scaling

### Horizontal Scaling

To scale the API service:

```bash
docker-compose up -d --scale api=3
```

### Monitoring Multiple Models

1. Create separate monitoring instances for each model
2. Use different Prometheus jobs for each model
3. Create model-specific Grafana dashboards

## Security

### Production Considerations

- Change default Grafana password
- Use HTTPS for all services
- Implement authentication for monitoring endpoints
- Use secrets management for sensitive tokens
- Configure network policies

### Access Control

```bash
# Change Grafana admin password
docker-compose exec grafana grafana-cli admin reset-admin-password newpassword
```

## Performance

### Optimization Tips

- Adjust Prometheus scrape intervals based on traffic
- Use data retention policies for long-term storage
- Configure Grafana caching for better performance
- Monitor resource usage of all services

### Resource Requirements

- **API**: 512MB RAM, 1 CPU
- **Prometheus**: 1GB RAM, 1 CPU
- **Grafana**: 256MB RAM, 0.5 CPU

## Support

For issues with the monitoring setup:

1. Check the logs: `docker-compose logs`
2. Verify service connectivity
3. Test individual components
4. Review configuration files

The monitoring setup provides comprehensive visibility into model performance and data quality, enabling proactive detection of issues and maintaining model reliability in production.
