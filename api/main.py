"""
SmartRecommendation FastAPI Application
Production-ready REST API for recommendation system
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import numpy as np
import pickle
import torch
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="SmartRecommendation API",
    description="Production-ready recommendation system API",
    version="1.0.0"
)

# Add CORS middleware (allow requests from any origin)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# MODELS & CLASSES
# ============================================================================

class SVDRecommender:
    """SVD Recommender (copy from Day 4)"""
    
    def __init__(self, n_factors=50):
        self.n_factors = n_factors
        self.svd_model = None
        self.user_factors = None
        self.product_factors = None
        self.mean_rating = None
        
    def fit(self, R):
        from sklearn.decomposition import TruncatedSVD
        self.mean_rating = np.nanmean(R[R > 0])
        R_centered = R.copy()
        R_centered[R_centered > 0] = R_centered[R_centered > 0] - self.mean_rating
        self.svd_model = TruncatedSVD(n_components=self.n_factors, random_state=42)
        U = self.svd_model.fit_transform(R_centered)
        V = self.svd_model.components_.T
        self.user_factors = U
        self.product_factors = V
        
    def predict(self, user_idx, product_idx=None):
        if product_idx is None:
            predictions = self.user_factors[user_idx] @ self.product_factors.T
        else:
            predictions = self.user_factors[user_idx] @ self.product_factors[product_idx]
        predictions = predictions + self.mean_rating
        predictions = np.clip(predictions, 0, 5)
        return predictions
    
    def recommend(self, user_idx, n_recommendations=10, exclude_seen=True, seen_items=None):
        predictions = self.predict(user_idx)
        if exclude_seen and seen_items is not None:
            predictions[seen_items] = -np.inf
        top_indices = np.argsort(predictions)[::-1][:n_recommendations]
        return top_indices


class MatrixFactorization:
    """MF Recommender (copy from Day 4)"""
    
    def __init__(self, n_factors=50, learning_rate=0.01, n_epochs=50, lambda_reg=0.01):
        self.n_factors = n_factors
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.lambda_reg = lambda_reg
        self.user_factors = None
        self.product_factors = None
        self.bias_user = None
        self.bias_product = None
        self.global_bias = None
        self.loss_history = []
        
    def fit(self, R):
        n_users, n_products = R.shape
        self.user_factors = np.random.normal(0, 0.1, (n_users, self.n_factors))
        self.product_factors = np.random.normal(0, 0.1, (self.n_factors, n_products))
        self.bias_user = np.zeros(n_users)
        self.bias_product = np.zeros(n_products)
        self.global_bias = np.mean(R[R > 0]) if np.any(R > 0) else 3.0
        
    def predict(self, user_idx, product_idx=None):
        if product_idx is None:
            predictions = (self.global_bias + self.bias_user[user_idx] + self.bias_product +
                          self.user_factors[user_idx] @ self.product_factors)
        else:
            predictions = (self.global_bias + self.bias_user[user_idx] + 
                          self.bias_product[product_idx] +
                          self.user_factors[user_idx] @ self.product_factors[:, product_idx])
        predictions = np.clip(predictions, 0, 5)
        return predictions
    
    def recommend(self, user_idx, n_recommendations=10, exclude_seen=True, seen_items=None):
        predictions = self.predict(user_idx)
        if exclude_seen and seen_items is not None:
            predictions[seen_items] = -np.inf
        top_indices = np.argsort(predictions)[::-1][:n_recommendations]
        return top_indices
import sys
sys.modules['__main__'].SVDRecommender = SVDRecommender
sys.modules['__main__'].MatrixFactorization = MatrixFactorization

class EnsembleRecommender:
    """Ensemble Recommender (SVD + MF only - NCF skipped)"""
    
    def __init__(self, svd_model, mf_model, ncf_model=None, weights=None):
        self.svd_model = svd_model
        self.mf_model = mf_model
        self.ncf_model = ncf_model
        
        if weights is None:
            # Use only SVD and MF weights
            self.weights = [0.5, 0.5]  # 50-50 split
        else:
            total = sum(weights[:2])  # Only count SVD and MF
            self.weights = [weights[0]/total, weights[1]/total]
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def predict(self, user_idx, product_idx=None):
        svd_pred = self.svd_model.predict(user_idx, product_idx)
        mf_pred = self.mf_model.predict(user_idx, product_idx)
        
        # Simple weighted average of SVD and MF
        ensemble_pred = (self.weights[0] * svd_pred + self.weights[1] * mf_pred)
        ensemble_pred = np.clip(ensemble_pred, 0, 5)
        
        return ensemble_pred
    
    def recommend(self, user_idx, n_recommendations=10, exclude_seen=True, seen_items=None):
        predictions = self.predict(user_idx)
        
        if exclude_seen and seen_items is not None:
            predictions[seen_items] = -np.inf
        
        top_indices = np.argsort(predictions)[::-1][:n_recommendations]
        
        return top_indices


# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class RecommendationRequest(BaseModel):
    """Request model for getting recommendations"""
    user_id: int = Field(..., description="User ID (0 to 999)")
    n_recommendations: int = Field(10, description="Number of recommendations (1-50)")
    model_type: str = Field("ensemble", description="Model to use: svd, mf, or ensemble")
    exclude_purchased: bool = Field(True, description="Exclude items user already interacted with")
    
    class Config:
        example = {
            "user_id": 5,
            "n_recommendations": 10,
            "model_type": "ensemble",
            "exclude_purchased": True
        }


class RatingPredictionRequest(BaseModel):
    """Request model for predicting ratings"""
    user_id: int = Field(..., description="User ID")
    product_id: int = Field(..., description="Product ID")
    model_type: str = Field("ensemble", description="Model to use")
    
    class Config:
        example = {
            "user_id": 5,
            "product_id": 10,
            "model_type": "ensemble"
        }


class Recommendation(BaseModel):
    """Single recommendation"""
    product_id: int
    predicted_rank: int
    confidence: float = Field(0.0, description="Confidence score (0-1)")


class RecommendationResponse(BaseModel):
    """Response model for recommendations"""
    user_id: int
    model_type: str
    recommendations: List[Recommendation]
    total_recommendations: int


class RatingPredictionResponse(BaseModel):
    """Response model for rating prediction"""
    user_id: int
    product_id: int
    predicted_rating: float = Field(..., ge=0, le=5)
    model_type: str


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    models_loaded: List[str]
    version: str


# ============================================================================
# GLOBAL MODEL STATE
# ============================================================================

class ModelState:
    """Store loaded models"""
    svd_model: Optional[SVDRecommender] = None
    mf_model: Optional[MatrixFactorization] = None
    ncf_model: Optional[torch.nn.Module] = None
    ensemble_model: Optional[EnsembleRecommender] = None
    train_matrix: Optional[np.ndarray] = None
    
    loaded_models: List[str] = []


model_state = ModelState()


# ============================================================================
# STARTUP & SHUTDOWN
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Load models on startup"""
    logger.info("Starting up application...")
    
    try:
        # Load train matrix
        model_state.train_matrix = np.load('data/processed/train_interaction_matrix.npy')
        
        # Load SVD model
        with open('models/svd_model.pkl', 'rb') as f:
            model_state.svd_model = pickle.load(f)
        model_state.loaded_models.append("svd")
        logger.info("✓ SVD model loaded")
        
        # Load MF model
        with open('models/mf_model.pkl', 'rb') as f:
            model_state.mf_model = pickle.load(f)
        model_state.loaded_models.append("mf")
        logger.info("✓ Matrix Factorization model loaded")
        
        # Load NCF model
        # Note: You'll need to define NeuralCollaborativeFiltering class in the API
        # For now, we skip NCF in the example
        logger.info("⚠ NCF model skipped (requires additional setup)")
        
        # Create ensemble
        if model_state.svd_model and model_state.mf_model:
            model_state.ensemble_model = EnsembleRecommender(
                svd_model=model_state.svd_model,
                mf_model=model_state.mf_model,
                ncf_model=None,  # Skip NCF for simplicity
                weights=[1, 1, 1]
            )
            model_state.loaded_models.append("ensemble")
            logger.info("✓ Ensemble model created")
        
        logger.info("✅ All models loaded successfully!")
        
    except Exception as e:
        logger.error(f"❌ Error loading models: {str(e)}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down application...")


# ============================================================================
# ENDPOINTS
# ============================================================================

@app.get("/", tags=["Info"])
async def root():
    """Root endpoint"""
    return {
        "message": "SmartRecommendation API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["Info"])
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if model_state.loaded_models else "unhealthy",
        models_loaded=model_state.loaded_models,
        version="1.0.0"
    )


@app.post("/recommendations", response_model=RecommendationResponse, tags=["Recommendations"])
async def get_recommendations(request: RecommendationRequest):
    """
    Get product recommendations for a user
    
    Parameters:
    - user_id: User ID (0-999)
    - n_recommendations: Number of recommendations (default: 10)
    - model_type: Which model to use (svd, mf, ensemble)
    - exclude_purchased: Exclude items user already interacted with
    """
    
    # Validate input
    if request.user_id < 0 or request.user_id >= model_state.train_matrix.shape[0]:
        raise HTTPException(
            status_code=400,
            detail=f"User ID must be between 0 and {model_state.train_matrix.shape[0] - 1}"
        )
    
    if request.n_recommendations < 1 or request.n_recommendations > 50:
        raise HTTPException(status_code=400, detail="n_recommendations must be between 1 and 50")
    
    if request.model_type not in model_state.loaded_models:
        raise HTTPException(
            status_code=400,
            detail=f"Model {request.model_type} not loaded. Available: {model_state.loaded_models}"
        )
    
    try:
        # Select model
        if request.model_type == "svd":
            model = model_state.svd_model
        elif request.model_type == "mf":
            model = model_state.mf_model
        elif request.model_type == "ensemble":
            model = model_state.ensemble_model
        else:
            raise ValueError(f"Unknown model: {request.model_type}")
        
        # Get seen items
        seen_items = np.where(model_state.train_matrix[request.user_id] > 0)[0] if request.exclude_purchased else None
        
        # Get recommendations
        recommendations_indices = model.recommend(
            user_idx=request.user_id,
            n_recommendations=request.n_recommendations,
            exclude_seen=request.exclude_purchased,
            seen_items=seen_items
        )
        
        # Create response
        recommendations = [
            Recommendation(
                product_id=int(idx),
                predicted_rank=rank + 1,
                confidence=float(1.0 - (rank / request.n_recommendations))  # Simple confidence metric
            )
            for rank, idx in enumerate(recommendations_indices)
        ]
        
        return RecommendationResponse(
            user_id=request.user_id,
            model_type=request.model_type,
            recommendations=recommendations,
            total_recommendations=len(recommendations)
        )
        
    except Exception as e:
        logger.error(f"Error generating recommendations: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.post("/predict-rating", response_model=RatingPredictionResponse, tags=["Predictions"])
async def predict_rating(request: RatingPredictionRequest):
    """
    Predict rating for a user-product pair
    
    Parameters:
    - user_id: User ID
    - product_id: Product ID
    - model_type: Which model to use
    """
    
    # Validate input
    if request.user_id < 0 or request.user_id >= model_state.train_matrix.shape[0]:
        raise HTTPException(status_code=400, detail="Invalid user ID")
    
    if request.product_id < 0 or request.product_id >= model_state.train_matrix.shape[1]:
        raise HTTPException(status_code=400, detail="Invalid product ID")
    
    if request.model_type not in model_state.loaded_models:
        raise HTTPException(status_code=400, detail=f"Model {request.model_type} not loaded")
    
    try:
        # Select model
        if request.model_type == "svd":
            model = model_state.svd_model
        elif request.model_type == "mf":
            model = model_state.mf_model
        elif request.model_type == "ensemble":
            model = model_state.ensemble_model
        else:
            raise ValueError(f"Unknown model: {request.model_type}")
        
        # Predict rating
        predicted_rating = float(model.predict(request.user_id, request.product_id))
        
        return RatingPredictionResponse(
            user_id=request.user_id,
            product_id=request.product_id,
            predicted_rating=predicted_rating,
            model_type=request.model_type
        )
        
    except Exception as e:
        logger.error(f"Error predicting rating: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.get("/models", tags=["Info"])
async def list_models():
    """List available models"""
    return {
        "available_models": model_state.loaded_models,
        "default_model": "ensemble",
        "description": "Use these models in the model_type parameter"
    }


# ============================================================================
# ERROR HANDLERS
# ============================================================================

from fastapi.responses import JSONResponse

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions"""
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "status_code": exc.status_code}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions"""
    logger.error(f"Unhandled error: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"error": str(exc), "status_code": 500}
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)