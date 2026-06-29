# SmartRecommendation: End-to-End ML Pipeline for E-Commerce

## Overview
A production-ready machine learning pipeline for personalized e-commerce product recommendations using collaborative filtering, content-based filtering, and deep learning approaches.

## Features
-  Multiple recommendation algorithms (SVD, NCF, Content-based, Hybrid)
-  Real-time inference API (FastAPI)
-  A/B testing framework
-  Model versioning and tracking (MLflow)
-  Docker containerization
-  Production deployment ready

## Project Structure
```
SmartRecommendation/
├── data/              # Raw, processed, and test data
├── notebooks/         # Jupyter notebooks for EDA and experiments
├── src/              # Source code modules
├── api/              # FastAPI application
├── tests/            # Unit and integration tests
├── config/           # Configuration files
├── experiments/      # MLflow tracking
└── docker/           # Docker files
```

## Setup Instructions

### 1. Create Virtual Environment
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate
```
2. Install Dependencies
```bash
pip install -r requirements.txt
```
3. Run Setup Test
```bash
python src/test_setup.py
```
4. Collect Data
```bash
python src/data_collection.py
```
Project Timeline
Week 1: Data collection and EDA
Week 2: Model development and comparison
Week 3-4: API development and deployment
Week 5-8: Production polish and optimization
Technologies Used
Python 3.10+
PyTorch for deep learning
Scikit-learn for ML algorithms
FastAPI for REST API
Docker for containerization
MLflow for experiment tracking
PostgreSQL for data storage
## 🚀 Live Demo

**API is now LIVE and publicly accessible!**

- **API Endpoint:** https://smartrecommendation-j9x5.onrender.com
- **Interactive Docs:** https://smartrecommendation-j9x5.onrender.com/docs
- **Health Check:** https://smartrecommendation-j9x5.onrender.com/health

### Try the API

```bash
# Get recommendations
curl -X POST "https://smartrecommendation-j9x5.onrender.com/recommendations" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": 5,
    "n_recommendations": 10,
    "model_type": "ensemble",
    "exclude_purchased": true
  }'

# Check health
curl https://smartrecommendation-j9x5.onrender.com/health
```
Author
Parnika Gupta
License
MIT
```
