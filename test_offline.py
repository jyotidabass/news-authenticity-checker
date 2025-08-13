#!/usr/bin/env python3
"""
Test script for the News Authenticity Checker
Tests both offline functionality and API integration
"""

import requests
import json
import time

# Configuration
BASE_URL = "http://localhost:5000"
TEST_NEWS = "Scientists discover that drinking coffee can cure all diseases and make you live forever."

def test_health_endpoint():
    """Test the health check endpoint"""
    print("🔍 Testing health endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health check passed: {data['status']}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

def test_api_status():
    """Test the API status endpoint"""
    print("\n🔍 Testing API status endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/api/status")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ API status retrieved successfully")
            print(f"   Active APIs: {data['enhancement_info']['active_apis']}/{data['enhancement_info']['total_apis']}")
            print(f"   Enhancement Level: {data['enhancement_info']['enhancement_percentage']:.1f}%")
            return True
        else:
            print(f"❌ API status failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ API status error: {e}")
        return False

def test_api_config():
    """Test the API configuration endpoint"""
    print("\n🔍 Testing API configuration endpoint...")
    try:
        # Test GET
        response = requests.get(f"{BASE_URL}/api/config")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ API config GET successful")
            print(f"   Available config keys: {list(data['config'].keys())}")
        else:
            print(f"❌ API config GET failed: {response.status_code}")
            return False
        
        # Test POST with dummy data
        test_config = {
            "GOOGLE_API_KEY": "test_key_123",
            "NEWS_API_KEY": "test_news_key"
        }
        
        response = requests.post(f"{BASE_URL}/api/config", json=test_config)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ API config POST successful: {data['message']}")
            return True
        else:
            print(f"❌ API config POST failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ API config error: {e}")
        return False

def test_authenticity_check():
    """Test the main authenticity checking functionality"""
    print("\n🔍 Testing authenticity check...")
    try:
        payload = {"news_text": TEST_NEWS}
        response = requests.post(f"{BASE_URL}/check_authenticity", json=payload)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Authenticity check successful")
            print(f"   Score: {data['authenticity_score']:.2f}")
            print(f"   Offline Mode: {data['offline_mode']}")
            print(f"   Enhanced with APIs: {data.get('enhanced_with_apis', False)}")
            
            if 'api_enhancements' in data and data['api_enhancements']:
                print(f"   API Enhancements: {', '.join(data['api_enhancements'])}")
            
            return True
        else:
            print(f"❌ Authenticity check failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Authenticity check error: {e}")
        return False

def test_offline_functionality():
    """Test that the app works completely offline"""
    print("\n🔍 Testing offline functionality...")
    try:
        # Test with a simple news text
        test_texts = [
            "This is a test news article.",
            "Scientists claim amazing discovery.",
            "Breaking news: incredible breakthrough in technology."
        ]
        
        for i, text in enumerate(test_texts, 1):
            payload = {"news_text": text}
            response = requests.post(f"{BASE_URL}/check_authenticity", json=payload)
            
            if response.status_code == 200:
                data = response.json()
                print(f"   ✅ Test {i}: Score {data['authenticity_score']:.2f}")
            else:
                print(f"   ❌ Test {i}: Failed with status {response.status_code}")
                return False
        
        print("✅ All offline tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Offline functionality error: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Starting News Authenticity Checker Tests")
    print("=" * 50)
    
    tests = [
        test_health_endpoint,
        test_api_status,
        test_api_config,
        test_authenticity_check,
        test_offline_functionality
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            time.sleep(0.5)  # Small delay between tests
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The app is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
