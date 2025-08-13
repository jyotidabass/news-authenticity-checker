import unittest
import json
from app import app, checker

class NewsAuthenticityCheckerTest(unittest.TestCase):
    
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True
    
    def test_health_endpoint(self):
        """Test health check endpoint"""
        response = self.app.get('/health')
        data = json.loads(response.data)
        
        self.assertEqual(response.status_code, 200)
        self.assertEqual(data['status'], 'healthy')
        self.assertIn('timestamp', data)
    
    def test_index_endpoint(self):
        """Test main page endpoint"""
        response = self.app.get('/')
        
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'News Authenticity Checker', response.data)
    
    def test_check_authenticity_empty_text(self):
        """Test authenticity check with empty text"""
        response = self.app.post('/check_authenticity',
                               data=json.dumps({'news_text': ''}),
                               content_type='application/json')
        
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertIn('error', data)
    
    def test_check_authenticity_valid_text(self):
        """Test authenticity check with valid text"""
        test_text = "The Earth orbits around the Sun, which is a fact verified by NASA."
        
        response = self.app.post('/check_authenticity',
                               data=json.dumps({'news_text': test_text}),
                               content_type='application/json')
        
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        
        # Check required fields
        self.assertIn('authenticity_score', data)
        self.assertIn('text_analysis', data)
        self.assertIn('similar_facts', data)
        self.assertIn('fact_check_results', data)
        self.assertIn('recommendations', data)
        
        # Check score range
        self.assertGreaterEqual(data['authenticity_score'], 0.0)
        self.assertLessEqual(data['authenticity_score'], 1.0)
        
        # Check text analysis
        self.assertIn('length', data['text_analysis'])
        self.assertIn('word_count', data['text_analysis'])
        self.assertIn('has_emotional_language', data['text_analysis'])
        self.assertIn('has_clickbait_patterns', data['text_analysis'])
        self.assertIn('has_credible_sources', data['text_analysis'])
        self.assertIn('sentiment', data['text_analysis'])

class NewsAuthenticityCheckerClassTest(unittest.TestCase):
    
    def setUp(self):
        self.checker = checker
    
    def test_text_analysis(self):
        """Test text characteristic analysis"""
        test_text = "This is a test text with some emotional language that might be shocking and amazing."
        
        analysis = self.checker.analyze_text_characteristics(test_text)
        
        self.assertIn('length', analysis)
        self.assertIn('word_count', analysis)
        self.assertIn('has_emotional_language', analysis)
        self.assertIn('has_clickbait_patterns', analysis)
        self.assertIn('has_credible_sources', analysis)
        self.assertIn('sentiment', analysis)
        
        # Test specific values
        self.assertGreater(analysis['length'], 0)
        self.assertGreater(analysis['word_count'], 0)
        self.assertIsInstance(analysis['has_emotional_language'], bool)
        self.assertIsInstance(analysis['has_clickbait_patterns'], bool)
        self.assertIsInstance(analysis['has_credible_sources'], bool)
        self.assertIn(analysis['sentiment'], ['positive', 'negative', 'neutral'])
    
    def test_emotional_language_detection(self):
        """Test emotional language detection"""
        emotional_text = "This is shocking and amazing news that will blow your mind!"
        normal_text = "This is a normal news article about current events."
        
        self.assertTrue(self.checker.detect_emotional_language(emotional_text))
        self.assertFalse(self.checker.detect_emotional_language(normal_text))
    
    def test_clickbait_detection(self):
        """Test clickbait pattern detection"""
        clickbait_text = "You won't believe what happens next!"
        normal_text = "New study shows interesting results."
        
        self.assertTrue(self.checker.detect_clickbait_patterns(clickbait_text))
        self.assertFalse(self.checker.detect_clickbait_patterns(normal_text))
    
    def test_credible_sources_detection(self):
        """Test credible sources detection"""
        credible_text = "According to university research and government studies, this is proven."
        normal_text = "Some people say this might be true."
        
        self.assertTrue(self.checker.detect_credible_sources(credible_text))
        self.assertFalse(self.checker.detect_credible_sources(normal_text))
    
    def test_sentiment_analysis(self):
        """Test sentiment analysis"""
        positive_text = "This is good and great news for everyone."
        negative_text = "This is bad and terrible news."
        neutral_text = "This is just information."
        
        self.assertEqual(self.checker.analyze_sentiment(positive_text), 'positive')
        self.assertEqual(self.checker.analyze_sentiment(negative_text), 'negative')
        self.assertEqual(self.checker.analyze_sentiment(neutral_text), 'neutral')

if __name__ == '__main__':
    unittest.main()
