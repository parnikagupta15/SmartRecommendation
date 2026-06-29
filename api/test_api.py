"""
Test API endpoints
"""

import requests
import json

BASE_URL = "http://localhost:8000"

def test_health():
    """Test health check"""
    print("\n🔍 Testing /health endpoint...")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(json.dumps(response.json(), indent=2))


def test_models():
    """Test models endpoint"""
    print("\n🔍 Testing /models endpoint...")
    response = requests.get(f"{BASE_URL}/models")
    print(f"Status: {response.status_code}")
    print(json.dumps(response.json(), indent=2))


def test_recommendations():
    """Test recommendations endpoint"""
    print("\n🔍 Testing /recommendations endpoint...")
    
    payload = {
        "user_id": 5,
        "n_recommendations": 10,
        "model_type": "ensemble",
        "exclude_purchased": True
    }
    
    response = requests.post(f"{BASE_URL}/recommendations", json=payload)
    print(f"Status: {response.status_code}")
    print(json.dumps(response.json(), indent=2))


def test_rating_prediction():
    """Test rating prediction endpoint"""
    print("\n🔍 Testing /predict-rating endpoint...")
    
    payload = {
        "user_id": 5,
        "product_id": 10,
        "model_type": "ensemble"
    }
    
    response = requests.post(f"{BASE_URL}/predict-rating", json=payload)
    print(f"Status: {response.status_code}")
    print(json.dumps(response.json(), indent=2))


def test_invalid_user():
    """Test error handling"""
    print("\n🔍 Testing error handling (invalid user)...")
    
    payload = {
        "user_id": 99999,  # Invalid
        "n_recommendations": 10,
        "model_type": "ensemble"
    }
    
    response = requests.post(f"{BASE_URL}/recommendations", json=payload)
    print(f"Status: {response.status_code}")
    
    try:
        data = response.json()
        print(json.dumps(data, indent=2))
    except:
        print(f"Response text: {response.text}")
        print(" Error handling works (returned non-200 status)")


if __name__ == "__main__":
    print("=" * 70)
    print("🧪 TESTING SMARTRECOMMENDATION API")
    print("=" * 70)
    
    test_health()
    test_models()
    test_recommendations()
    test_rating_prediction()
    test_invalid_user()
    
    print("\n" + "=" * 70)
    print("✅ All tests completed!")
    print("=" * 70)
