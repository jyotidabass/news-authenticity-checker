import requests
import json
import time

# Configuration
BASE_URL = "http://localhost:5000"
TEST_NEWS = "Scientists discover that drinking coffee can cure all diseases and make you live forever."

def test_health_endpoint():
    """Test the health endpoint"""
    print("🏥 Testing health endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            print("✅ Health endpoint working")
            return True
        else:
            print(f"❌ Health endpoint failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health endpoint error: {e}")
        return False

def test_api_status():
    """Test the API status endpoint"""
    print("\n📊 Testing API status endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/api/status")
        if response.status_code == 200:
            data = response.json()
            print("✅ API status endpoint working")
            print(f"   Active APIs: {data['enhancement_info']['active_apis']}/{data['enhancement_info']['total_apis']}")
            print(f"   Enhancement: {data['enhancement_info']['enhancement_percentage']:.1f}%")
            
            # Show individual API status
            for api in data['available_apis']:
                status = "🟢 Active" if api['status'] else "🔴 Inactive"
                print(f"   {api['name']}: {status}")
            
            return True
        else:
            print(f"❌ API status failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ API status error: {e}")
        return False

def test_api_config():
    """Test the API configuration endpoint"""
    print("\n⚙️ Testing API configuration endpoint...")
    
    # Test GET
    try:
        response = requests.get(f"{BASE_URL}/api/config")
        if response.status_code == 200:
            data = response.json()
            print("✅ GET config endpoint working")
            print(f"   Available config keys: {list(data['config'].keys())}")
        else:
            print(f"❌ GET config failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ GET config error: {e}")
        return False
    
    # Test POST with dummy data
    try:
        test_config = {
            "GOOGLE_API_KEY": "test_key_123",
            "NEWS_API_KEY": "test_news_key",
            "OPENAI_API_KEY": "test_openai_key",
            "PINECONE_API_KEY": "test_pinecone_key",
            "PINECONE_ENVIRONMENT": "test-env"
        }
        
        response = requests.post(f"{BASE_URL}/api/config", json=test_config)
        if response.status_code == 200:
            data = response.json()
            print("✅ POST config endpoint working")
            print(f"   Configuration updated: {data['status']}")
        else:
            print(f"❌ POST config failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ POST config error: {e}")
        return False
    
    return True

def test_authenticity_check():
    """Test the main authenticity checking endpoint"""
    print("\n🔍 Testing authenticity check endpoint...")
    try:
        response = requests.post(f"{BASE_URL}/check_authenticity", 
                               json={"news_text": TEST_NEWS})
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Authenticity check working")
            print(f"   Authenticity Score: {data['authenticity_score']:.2f}")
            print(f"   Offline Mode: {data['offline_mode']}")
            print(f"   Enhanced with APIs: {data['enhanced_with_apis']}")
            print(f"   API Enhancements: {data['api_enhancements']}")
            
            # Check if we have fact check results
            if 'fact_check_results' in data:
                fc = data['fact_check_results']
                print(f"   Fact Check Available: {fc.get('fact_check_available', False)}")
                print(f"   Verdict: {fc.get('verdict', 'None')}")
                print(f"   Confidence: {fc.get('confidence', 0.0)}")
            
            return True
        else:
            print(f"❌ Authenticity check failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Authenticity check error: {e}")
        return False

def test_offline_functionality():
    """Test that the app works completely offline"""
    print("\n🆓 Testing offline functionality...")
    try:
        # Test with a simple news text
        simple_news = "The sky is blue."
        response = requests.post(f"{BASE_URL}/check_authenticity", 
                               json={"news_text": simple_news})
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Offline functionality working")
            print(f"   Score calculated: {data['authenticity_score']:.2f}")
            print(f"   Text analysis available: {'text_analysis' in data}")
            print(f"   Similar facts found: {len(data.get('similar_facts', []))}")
            print(f"   Recommendations: {len(data.get('recommendations', []))}")
            return True
        else:
            print(f"❌ Offline functionality failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Offline functionality error: {e}")
        return False

def test_api_integration():
    """Test that APIs enhance results when available"""
    print("\n🚀 Testing API integration...")
    
    # First check current API status
    try:
        response = requests.get(f"{BASE_URL}/api/status")
        if response.status_code == 200:
            data = response.json()
            active_apis = [api['name'] for api in data['available_apis'] if api['status']]
            
            if active_apis:
                print(f"✅ Found active APIs: {', '.join(active_apis)}")
                print("   Testing enhanced results...")
                
                # Test with news text
                response = requests.post(f"{BASE_URL}/check_authenticity", 
                                       json={"news_text": TEST_NEWS})
                
                if response.status_code == 200:
                    data = response.json()
                    if data['enhanced_with_apis']:
                        print("✅ API enhancement working")
                        print(f"   APIs used: {data['api_enhancements']}")
                        return True
                    else:
                        print("⚠️ APIs available but not enhancing results")
                        return False
                else:
                    print(f"❌ API test failed: {response.status_code}")
                    return False
            else:
                print("ℹ️ No APIs currently active - this is normal for testing")
                return True
        else:
            print(f"❌ Could not check API status: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ API integration test error: {e}")
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
        test_offline_functionality,
        test_api_integration
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            print()
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
            print()
    
    print("=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The app is working correctly.")
    else:
        print("⚠️ Some tests failed. Check the output above for details.")
    
    print("\n💡 Tips:")
    print("   - Make sure the Flask app is running on localhost:5000")
    print("   - Add API keys through the web interface for enhanced results")
    print("   - The app works completely offline by default")

if __name__ == "__main__":
    main()
