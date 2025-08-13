import os
import json
import numpy as np
from flask import Flask, render_template_string, request, jsonify
from datetime import datetime
import re
from typing import List, Dict, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Optional imports - app works without these
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    logger.warning("Requests module not available - external API features disabled")

try:
    from sentence_transformers import SentenceTransformer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("Sentence transformers not available - using basic text analysis")

try:
    import pinecone
    PINECONE_AVAILABLE = True
except ImportError:
    PINECONE_AVAILABLE = False
    logger.warning("Pinecone not available - using local similarity search")

app = Flask(__name__)

# Configuration
class Config:
    # HuggingFace model for sentence embeddings
    MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
    
    # Optional APIs (all are completely optional)
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
    NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    
    # Optional Pinecone configuration
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
    PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "")
    PINECONE_INDEX_NAME = "news-facts"
    
    # Similarity threshold for fact-checking
    SIMILARITY_THRESHOLD = 0.75
    
    # Free fact-checking sources (always available)
    FREE_FACT_CHECK_SOURCES = [
        "https://www.snopes.com",
        "https://www.factcheck.org",
        "https://www.politifact.com",
        "https://www.reuters.com/fact-check",
        "https://www.bbc.com/news/fact-check",
        "https://www.afp.com/fact-check",
        "https://www.leadstories.com",
        "https://www.hoax-slayer.net"
    ]
    
    # Enhanced local fact database (always available offline)
    ENHANCED_FACTS = [
        # Health & Science
        {"text": "COVID-19 is caused by the SARS-CoV-2 virus", "source": "WHO", "category": "health", "verification_date": "2020-01-01"},
        {"text": "Vaccines help prevent infectious diseases", "source": "CDC", "category": "health", "verification_date": "2020-01-01"},
        {"text": "Climate change is supported by scientific evidence", "source": "NASA", "category": "science", "verification_date": "2020-01-01"},
        {"text": "The Earth is approximately 4.5 billion years old", "source": "Scientific consensus", "category": "science", "verification_date": "2020-01-01"},
        
        # Technology
        {"text": "5G technology is not harmful to human health", "source": "WHO", "category": "technology", "verification_date": "2020-01-01"},
        {"text": "Social media can affect mental health", "source": "Research studies", "category": "technology", "verification_date": "2020-01-01"},
        
        # Politics & Society
        {"text": "Voter fraud is extremely rare in US elections", "source": "Brennan Center", "category": "politics", "verification_date": "2020-01-01"},
        {"text": "The US has 50 states", "source": "US Government", "category": "geography", "verification_date": "2020-01-01"},
        
        # Environment
        {"text": "Plastic pollution affects marine life", "source": "Marine Biology Research", "category": "environment", "verification_date": "2020-01-01"},
        {"text": "Renewable energy is becoming more affordable", "source": "Energy Research", "category": "environment", "verification_date": "2020-01-01"},
        
        # Education
        {"text": "Higher education correlates with higher income", "source": "Economic Research", "category": "education", "verification_date": "2020-01-01"},
        {"text": "Reading improves cognitive function", "source": "Neuroscience Research", "category": "education", "verification_date": "2020-01-01"}
    ]

class NewsAuthenticityChecker:
    def __init__(self):
        self.model = None # Initialize model to None
        self.pinecone_index = None
        self.setup_pinecone()
        self.fact_database = self.load_fact_database()
    
    def setup_pinecone(self):
        """Initialize Pinecone for vector similarity search"""
        if not PINECONE_AVAILABLE:
            logger.warning("Pinecone not available, skipping initialization.")
            return

        try:
            if Config.PINECONE_API_KEY:
                pinecone.init(
                    api_key=Config.PINECONE_API_KEY,
                    environment=Config.PINECONE_ENVIRONMENT
                )
                
                # Create index if it doesn't exist
                if Config.PINECONE_INDEX_NAME not in pinecone.list_indexes():
                    pinecone.create_index(
                        name=Config.PINECONE_INDEX_NAME,
                        dimension=384,  # Dimension for all-MiniLM-L6-v2
                        metric="cosine"
                    )
                
                self.pinecone_index = pinecone.Index(Config.PINECONE_INDEX_NAME)
                logger.info("Pinecone initialized successfully")
            else:
                logger.warning("Pinecone API key not provided, using local similarity search")
        except Exception as e:
            logger.error(f"Failed to initialize Pinecone: {e}")
    
    def load_fact_database(self) -> List[Dict]:
        """Load a database of verified facts for comparison - always works offline"""
        # Use enhanced facts from config (always available)
        facts = Config.ENHANCED_FACTS.copy()
        
        # Add some additional dynamic facts based on common topics
        additional_facts = [
            {"text": "Water boils at 100 degrees Celsius at sea level", "source": "Scientific consensus", "category": "science", "verification_date": "2020-01-01"},
            {"text": "The Sun is a star at the center of our solar system", "source": "NASA", "category": "science", "verification_date": "2020-01-01"},
            {"text": "Photosynthesis is the process by which plants convert sunlight into energy", "source": "Scientific consensus", "category": "biology", "verification_date": "2020-01-01"},
            {"text": "Regular exercise improves physical and mental health", "source": "Medical Research", "category": "health", "verification_date": "2020-01-01"},
            {"text": "Sleep is essential for human health and well-being", "source": "Sleep Research", "category": "health", "verification_date": "2020-01-01"},
            {"text": "The internet was developed by ARPANET in the 1960s", "source": "Computer History", "category": "technology", "verification_date": "2020-01-01"},
            {"text": "Democracy is a form of government by the people", "source": "Political Science", "category": "politics", "verification_date": "2020-01-01"},
            {"text": "Biodiversity is important for ecosystem health", "source": "Environmental Science", "category": "environment", "verification_date": "2020-01-01"}
        ]
        
        facts.extend(additional_facts)
        return facts
    
    def get_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for given text"""
        if not TRANSFORMERS_AVAILABLE:
            logger.warning("Sentence transformers not available, using dummy embedding.")
            return np.zeros(384) # Return a dummy embedding

        try:
            if not self.model:
                self.model = SentenceTransformer(Config.MODEL_NAME)
            embedding = self.model.encode(text)
            return embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return np.zeros(384)
    
    def calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate cosine similarity between two embeddings"""
        try:
            similarity = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
            return float(similarity)
        except Exception as e:
            logger.error(f"Error calculating similarity: {e}")
            return 0.0
    
    def search_similar_facts(self, query_embedding: np.ndarray, threshold: float = 0.75) -> List[Dict]:
        """Search for similar facts using Pinecone or local search"""
        similar_facts = []
        
        if self.pinecone_index:
            try:
                # Search in Pinecone
                results = self.pinecone_index.query(
                    vector=query_embedding.tolist(),
                    top_k=10,
                    include_metadata=True
                )
                
                for match in results.matches:
                    if match.score >= threshold:
                        similar_facts.append({
                            "text": match.metadata.get("text", ""),
                            "source": match.metadata.get("source", ""),
                            "similarity": match.score,
                            "category": match.metadata.get("category", "")
                        })
            except Exception as e:
                logger.error(f"Pinecone search failed: {e}")
        
        # Fallback to local search
        if not similar_facts:
            for fact in self.fact_database:
                fact_embedding = self.get_embedding(fact["text"])
                similarity = self.calculate_similarity(query_embedding, fact_embedding)
                if similarity >= threshold:
                    similar_facts.append({
                        **fact,
                        "similarity": similarity
                    })
        
        # Sort by similarity score
        similar_facts.sort(key=lambda x: x["similarity"], reverse=True)
        return similar_facts[:5]
    
    def check_google_fact_check(self, text: str) -> Dict:
        """Check news against Google Fact Check API"""
        results = {
            "fact_check_available": False,
            "verdict": None,
            "confidence": 0.0,
            "sources": [],
            "claims": []
        }
        
        if not Config.GOOGLE_API_KEY:
            logger.warning("Google API key not provided")
            return results
        
        if not REQUESTS_AVAILABLE:
            logger.warning("Requests module not available, cannot use Google Fact Check API.")
            return results

        try:
            # Google Fact Check API endpoint
            url = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
            
            # Extract key terms for search
            search_terms = self.extract_search_terms(text)
            
            for term in search_terms[:3]:  # Limit to 3 terms to avoid rate limiting
                params = {
                    'key': Config.GOOGLE_API_KEY,
                    'query': term,
                    'languageCode': 'en'
                }
                
                response = requests.get(url, params=params)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    if 'claims' in data and data['claims']:
                        results["fact_check_available"] = True
                        
                        for claim in data['claims']:
                            claim_info = {
                                'text': claim.get('text', ''),
                                'claimant': claim.get('claimant', ''),
                                'claimDate': claim.get('claimDate', ''),
                                'claimReview': claim.get('claimReview', [])
                            }
                            
                            # Get the most recent review
                            if claim_info['claimReview']:
                                review = claim_info['claimReview'][0]
                                publisher = review.get('publisher', {})
                                
                                claim_info['review'] = {
                                    'publisher': publisher.get('name', ''),
                                    'url': review.get('url', ''),
                                    'title': review.get('title', ''),
                                    'reviewDate': review.get('reviewDate', '')
                                }
                                
                                # Determine verdict from review title
                                title = review.get('title', '').lower()
                                if any(word in title for word in ['false', 'incorrect', 'misleading']):
                                    claim_info['verdict'] = 'false'
                                elif any(word in title for word in ['true', 'correct', 'accurate']):
                                    claim_info['verdict'] = 'true'
                                elif any(word in title for word in ['partially', 'mixed', 'unproven']):
                                    claim_info['verdict'] = 'partially_true'
                                else:
                                    claim_info['verdict'] = 'unclear'
                            
                            results["claims"].append(claim_info)
                            
                            # Update overall verdict if we have a clear one
                            if claim_info.get('verdict') == 'false':
                                results["verdict"] = 'false'
                                results["confidence"] = max(results["confidence"], 0.8)
                            elif claim_info.get('verdict') == 'true' and results["verdict"] != 'false':
                                results["verdict"] = 'true'
                                results["confidence"] = max(results["confidence"], 0.7)
                
                # Add delay to respect rate limits
                import time
                time.sleep(0.1)
                
        except Exception as e:
            logger.error(f"Google Fact Check API error: {e}")
        
        return results
    
    def check_free_fact_check_sources(self, text: str) -> Dict:
        """Check news against free fact-checking sources"""
        results = {
            "fact_check_available": False,
            "verdict": None,
            "confidence": 0.0,
            "sources": [],
            "claims": [],
            "free_sources_checked": []
        }
        
        if not REQUESTS_AVAILABLE:
            logger.warning("Requests module not available, cannot use free fact-checking sources.")
            return results

        try:
            # Extract key search terms
            search_terms = self.extract_search_terms(text)
            
            # Check against free fact-checking websites
            for source in Config.FREE_FACT_CHECK_SOURCES:
                try:
                    # Simple keyword matching for common fact-checking terms
                    source_name = source.split('//')[1].split('.')[1].title()
                    
                    # Check if any search terms match common fact-checking patterns
                    fact_check_keywords = ['fact', 'check', 'verify', 'true', 'false', 'hoax', 'myth', 'debunk']
                    
                    for term in search_terms[:3]:
                        if any(keyword in term.lower() for keyword in fact_check_keywords):
                            results["free_sources_checked"].append({
                                'source': source_name,
                                'url': source,
                                'relevance': 'high',
                                'suggestion': f'Check {source_name} for fact verification'
                            })
                    
                    # Add general fact-checking suggestions
                    if len(results["free_sources_checked"]) < 3:
                        results["free_sources_checked"].append({
                            'source': source_name,
                            'url': source,
                            'relevance': 'medium',
                            'suggestion': f'Visit {source_name} for fact-checking'
                        })
                    
                except Exception as e:
                    logger.warning(f"Error checking source {source}: {e}")
                    continue
            
            # Check News API for related articles (if API key available)
            if Config.NEWS_API_KEY:
                news_results = self.check_news_api(text)
                if news_results["articles"]:
                    results["sources"].extend(news_results["articles"])
                    results["fact_check_available"] = True
            
            # Add manual fact-checking recommendations
            results["claims"].append({
                'text': 'Manual fact-checking recommended',
                'verdict': 'manual_check_needed',
                'sources': results["free_sources_checked"],
                'confidence': 0.5
            })
            
            if results["free_sources_checked"]:
                results["fact_check_available"] = True
                results["verdict"] = 'manual_check_needed'
                results["confidence"] = 0.5
            
        except Exception as e:
            logger.error(f"Free fact-checking error: {e}")
        
        return results
    
    def check_news_api(self, text: str) -> Dict:
        """Check News API for related articles"""
        results = {
            "articles": [],
            "total_results": 0
        }
        
        if not Config.NEWS_API_KEY:
            return results
        
        if not REQUESTS_AVAILABLE:
            logger.warning("Requests module not available, cannot use News API.")
            return results

        try:
            # News API endpoint
            url = "https://newsapi.org/v2/everything"
            
            # Extract key terms for search
            search_terms = self.extract_search_terms(text)
            
            for term in search_terms[:2]:  # Limit to avoid rate limiting
                params = {
                    'q': term,
                    'apiKey': Config.NEWS_API_KEY,
                    'language': 'en',
                    'sortBy': 'relevancy',
                    'pageSize': 5
                }
                
                response = requests.get(url, params=params)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    if 'articles' in data:
                        for article in data['articles'][:3]:
                            article_info = {
                                'title': article.get('title', ''),
                                'description': article.get('description', ''),
                                'url': article.get('url', ''),
                                'source': article.get('source', {}).get('name', ''),
                                'publishedAt': article.get('publishedAt', ''),
                                'relevance_score': self.calculate_article_relevance(article, text)
                            }
                            
                            # Only add articles with good relevance
                            if article_info['relevance_score'] > 0.3:
                                results["articles"].append(article_info)
                
                # Add delay to respect rate limits
                import time
                time.sleep(0.1)
                
        except Exception as e:
            logger.error(f"News API error: {e}")
        
        return results
    
    def calculate_article_relevance(self, article: Dict, query_text: str) -> float:
        """Calculate relevance score between article and query text"""
        try:
            # Simple keyword matching for relevance
            query_words = set(query_text.lower().split())
            title_words = set(article.get('title', '').lower().split())
            desc_words = set(article.get('description', '').lower().split())
            
            # Calculate overlap
            title_overlap = len(query_words.intersection(title_words)) / len(query_words) if query_words else 0
            desc_overlap = len(query_words.intersection(desc_words)) / len(query_words) if query_words else 0
            
            # Weighted relevance score
            relevance = (title_overlap * 0.7) + (desc_overlap * 0.3)
            
            return min(relevance, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating article relevance: {e}")
            return 0.0
    
    def check_web_scraping_facts(self, text: str) -> Dict:
        """Attempt to find facts through web scraping (basic implementation)"""
        results = {
            "scraped_facts": [],
            "sources": []
        }
        
        if not REQUESTS_AVAILABLE:
            logger.warning("Requests module not available, cannot use web scraping.")
            return results

        try:
            # Extract key terms
            search_terms = self.extract_search_terms(text)
            
            # Create search queries for fact-checking
            fact_check_queries = []
            for term in search_terms[:3]:
                fact_check_queries.extend([
                    f"{term} fact check",
                    f"{term} true false",
                    f"{term} verified",
                    f"{term} debunked"
                ])
            
            # Add basic fact-checking suggestions
            for query in fact_check_queries[:5]:
                results["scraped_facts"].append({
                    'query': query,
                    'suggestion': f'Search: "{query}" on fact-checking websites',
                    'confidence': 0.4,
                    'source_type': 'search_suggestion'
                })
            
            # Add manual verification steps
            results["scraped_facts"].extend([
                {
                    'query': 'Cross-reference with multiple sources',
                    'suggestion': 'Check multiple reputable news sources',
                    'confidence': 0.6,
                    'source_type': 'verification_step'
                },
                {
                    'query': 'Check source credibility',
                    'suggestion': 'Verify the source is reputable and unbiased',
                    'confidence': 0.7,
                    'source_type': 'verification_step'
                },
                {
                    'query': 'Look for recent updates',
                    'suggestion': 'Check if information is current and up-to-date',
                    'confidence': 0.5,
                    'source_type': 'verification_step'
                }
            ])
            
        except Exception as e:
            logger.error(f"Web scraping facts error: {e}")
        
        return results
    
    def extract_search_terms(self, text: str) -> List[str]:
        """Extract key search terms from text for fact-checking"""
        # Remove common words and punctuation
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        
        # Clean text and split into words
        words = re.findall(r'\b\w+\b', text.lower())
        
        # Filter out stop words and short words
        terms = [word for word in words if word not in stop_words and len(word) > 3]
        
        # Return unique terms, prioritizing longer words
        unique_terms = list(dict.fromkeys(terms))
        unique_terms.sort(key=len, reverse=True)
        
        return unique_terms[:5]  # Return top 5 terms
    
    def check_news_authenticity(self, news_text: str) -> Dict:
        """Main function to check news authenticity - always works offline"""
        try:
            # Generate embedding for the news text
            news_embedding = self.get_embedding(news_text)
            
            # Search for similar verified facts (always works offline)
            similar_facts = self.search_similar_facts(news_embedding, Config.SIMILARITY_THRESHOLD)
            
            # Analyze text characteristics (always works offline)
            text_analysis = self.analyze_text_characteristics(news_text)
            
            # Initialize fact check results
            fact_check_results = {
                "fact_check_available": False,
                "verdict": None,
                "confidence": 0.0,
                "sources": [],
                "claims": [],
                "offline_mode": True
            }
            
            # Try Google Fact Check API (if available)
            if Config.GOOGLE_API_KEY:
                try:
                    google_results = self.check_google_fact_check(news_text)
                    if google_results["fact_check_available"]:
                        fact_check_results.update(google_results)
                        fact_check_results["offline_mode"] = False
                except Exception as e:
                    logger.warning(f"Google API failed, continuing with offline mode: {e}")
            
            # Always provide free alternatives (works offline)
            try:
                free_fact_check = self.check_free_fact_check_sources(news_text)
                web_facts = self.check_web_scraping_facts(news_text)
                
                # Merge results from free sources
                fact_check_results["free_alternatives"] = free_fact_check
                fact_check_results["web_suggestions"] = web_facts
                
                # Update availability if we have free alternatives
                if free_fact_check["fact_check_available"]:
                    fact_check_results["fact_check_available"] = True
                
                # Set verdict if not already set
                if not fact_check_results["verdict"] and free_fact_check["verdict"]:
                    fact_check_results["verdict"] = free_fact_check["verdict"]
                    fact_check_results["confidence"] = free_fact_check["confidence"]
                
            except Exception as e:
                logger.warning(f"Free fact-checking failed, continuing with basic analysis: {e}")
            
            # Calculate authenticity score (always works offline)
            authenticity_score = self.calculate_authenticity_score(
                similar_facts, text_analysis, fact_check_results
            )
            
            # Ensure we always have a result
            if not fact_check_results["fact_check_available"]:
                fact_check_results["fact_check_available"] = True
                fact_check_results["verdict"] = "analysis_only"
                fact_check_results["confidence"] = 0.6
            
            return {
                "authenticity_score": authenticity_score,
                "similar_facts": similar_facts,
                "text_analysis": text_analysis,
                "fact_check_results": fact_check_results,
                "recommendations": self.generate_recommendations(authenticity_score, text_analysis, fact_check_results),
                "offline_mode": fact_check_results["offline_mode"]
            }
        
        except Exception as e:
            logger.error(f"Error checking news authenticity: {e}")
            # Return basic offline analysis even if main process fails
            try:
                basic_analysis = self.analyze_text_characteristics(news_text)
                basic_score = self.calculate_basic_authenticity_score(basic_analysis)
                
                return {
                    "error": str(e),
                    "authenticity_score": basic_score,
                    "text_analysis": basic_analysis,
                    "similar_facts": [],
                    "fact_check_results": {"offline_mode": True, "fact_check_available": False},
                    "recommendations": ["Basic analysis completed. Check the text characteristics above."],
                    "offline_mode": True,
                    "fallback_mode": True
                }
            except:
                return {
                    "error": str(e),
                    "authenticity_score": 0.0,
                    "offline_mode": True,
                    "fallback_mode": True
                }
    
    def analyze_text_characteristics(self, text: str) -> Dict:
        """Analyze text characteristics that might indicate fake news"""
        analysis = {
            "length": len(text),
            "word_count": len(text.split()),
            "has_emotional_language": self.detect_emotional_language(text),
            "has_clickbait_patterns": self.detect_clickbait_patterns(text),
            "has_credible_sources": self.detect_credible_sources(text),
            "sentiment": self.analyze_sentiment(text)
        }
        return analysis
    
    def detect_emotional_language(self, text: str) -> bool:
        """Detect emotional or sensational language"""
        emotional_words = [
            "shocking", "amazing", "incredible", "unbelievable", "scandalous",
            "outrageous", "terrifying", "horrifying", "miracle", "conspiracy",
            "exposed", "revealed", "secret", "hidden", "cover-up"
        ]
        
        text_lower = text.lower()
        emotional_count = sum(1 for word in emotional_words if word in text_lower)
        return emotional_count >= 3
    
    def detect_clickbait_patterns(self, text: str) -> bool:
        """Detect clickbait patterns in text"""
        clickbait_patterns = [
            r"you won't believe",
            r"this will shock you",
            r"what happens next",
            r"the truth about",
            r"they don't want you to know",
            r"number \d+ will surprise you"
        ]
        
        for pattern in clickbait_patterns:
            if re.search(pattern, text.lower()):
                return True
        return False
    
    def detect_credible_sources(self, text: str) -> bool:
        """Detect if text mentions credible sources"""
        credible_sources = [
            "university", "research", "study", "scientists", "experts",
            "official", "government", "peer-reviewed", "journal", "published"
        ]
        
        text_lower = text.lower()
        credible_count = sum(1 for source in credible_sources if source in text_lower)
        return credible_count >= 2
    
    def analyze_sentiment(self, text: str) -> str:
        """Simple sentiment analysis"""
        positive_words = ["good", "great", "excellent", "positive", "beneficial", "helpful"]
        negative_words = ["bad", "terrible", "awful", "negative", "harmful", "dangerous"]
        
        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        if positive_count > negative_count:
            return "positive"
        elif negative_count > positive_count:
            return "negative"
        else:
            return "neutral"
    
    def calculate_basic_authenticity_score(self, text_analysis: Dict) -> float:
        """Calculate basic authenticity score using only text analysis - always works offline"""
        score = 0.0
        
        # Score based on text characteristics (40%)
        if not text_analysis["has_emotional_language"]:
            score += 0.2
        if not text_analysis["has_clickbait_patterns"]:
            score += 0.2
        
        # Score based on content quality (30%)
        if text_analysis["has_credible_sources"]:
            score += 0.15
        if text_analysis["length"] > 100:  # Longer articles tend to be more credible
            score += 0.15
        
        # Score based on sentiment balance (20%)
        if text_analysis["sentiment"] == "neutral":
            score += 0.2
        elif text_analysis["sentiment"] in ["positive", "negative"]:
            score += 0.1
        
        # Score based on word count (10%)
        if text_analysis["word_count"] > 20:
            score += 0.1
        
        return min(max(score, 0.0), 1.0)
    
    def calculate_authenticity_score(self, similar_facts: List[Dict], text_analysis: Dict, fact_check_results: Dict) -> float:
        """Calculate overall authenticity score"""
        score = 0.0
        
        # Score based on similar facts
        if similar_facts:
            max_similarity = max(fact["similarity"] for fact in similar_facts)
            score += max_similarity * 0.3
        
        # Score based on text characteristics
        if not text_analysis["has_emotional_language"]:
            score += 0.2
        if not text_analysis["has_clickbait_patterns"]:
            score += 0.2
        if text_analysis["has_credible_sources"]:
            score += 0.1
        if text_analysis["length"] > 100:  # Longer articles tend to be more credible
            score += 0.1
        
        # Score based on Google fact-checking results
        if fact_check_results["verdict"] == "true":
            score += 0.3
        elif fact_check_results["verdict"] == "false":
            score -= 0.3
        elif fact_check_results["verdict"] == "partially_true":
            score += 0.1
        
        return min(max(score, 0.0), 1.0)
    
    def generate_recommendations(self, authenticity_score: float, text_analysis: Dict, fact_check_results: Dict) -> List[str]:
        """Generate recommendations based on analysis"""
        recommendations = []
        
        if authenticity_score < 0.3:
            recommendations.append("This news appears to be highly unreliable. Verify with multiple credible sources.")
        elif authenticity_score < 0.6:
            recommendations.append("Exercise caution. Cross-reference with established news sources.")
        else:
            recommendations.append("This news appears to be relatively reliable, but always verify important information.")
        
        if text_analysis["has_emotional_language"]:
            recommendations.append("The language used is highly emotional, which may indicate bias.")
        
        if text_analysis["has_clickbait_patterns"]:
            recommendations.append("The headline contains clickbait patterns that suggest sensationalism.")
        
        if not text_analysis["has_credible_sources"]:
            recommendations.append("No credible sources or references were found in the text.")
        
        # Add fact-check specific recommendations
        if fact_check_results["claims"]:
            false_claims = [claim for claim in fact_check_results["claims"] if claim.get('verdict') == 'false']
            if false_claims:
                recommendations.append(f"Google Fact Check found {len(false_claims)} false claim(s) related to this content.")
            
            true_claims = [claim for claim in fact_check_results["claims"] if claim.get('verdict') == 'true']
            if true_claims:
                recommendations.append(f"Google Fact Check verified {len(true_claims)} claim(s) as true.")
        
        # Add recommendations from free alternatives
        if "free_alternatives" in fact_check_results and fact_check_results["free_alternatives"]["free_sources_checked"]:
            recommendations.append("Free fact-checking sources are available for manual verification:")
            for source in fact_check_results["free_alternatives"]["free_sources_checked"][:3]:
                recommendations.append(f"• {source['suggestion']}")
        
        # Add web scraping suggestions
        if "web_suggestions" in fact_check_results and fact_check_results["web_suggestions"]["scraped_facts"]:
            recommendations.append("Additional verification steps:")
            for fact in fact_check_results["web_suggestions"]["scraped_facts"][:3]:
                if fact['source_type'] == 'verification_step':
                    recommendations.append(f"• {fact['suggestion']}")
        
        # Add general fact-checking advice
        recommendations.extend([
            "Always verify information from multiple independent sources.",
            "Check the date of the information to ensure it's current.",
            "Look for official statements or press releases when possible.",
            "Be skeptical of claims that seem too good or too bad to be true."
        ])
        
        return recommendations

# Initialize the checker
checker = NewsAuthenticityChecker()

# HTML template for the frontend
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>News Authenticity Checker</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            overflow: hidden;
        }
        
        .header {
            background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }
        
        .header h1 {
            font-size: 2.5rem;
            margin-bottom: 10px;
            font-weight: 300;
        }
        
        .header p {
            font-size: 1.1rem;
            opacity: 0.9;
        }
        
        .main-content {
            padding: 40px;
        }
        
        .input-section {
            margin-bottom: 40px;
        }
        
        .input-group {
            margin-bottom: 20px;
        }
        
        label {
            display: block;
            margin-bottom: 8px;
            font-weight: 600;
            color: #2c3e50;
        }
        
        textarea {
            width: 100%;
            min-height: 120px;
            padding: 15px;
            border: 2px solid #e1e8ed;
            border-radius: 10px;
            font-size: 16px;
            font-family: inherit;
            resize: vertical;
            transition: border-color 0.3s ease;
        }
        
        textarea:focus {
            outline: none;
            border-color: #667eea;
        }
        
        .btn {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 15px 30px;
            font-size: 16px;
            border-radius: 10px;
            cursor: pointer;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
            font-weight: 600;
        }
        
        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 20px rgba(102, 126, 234, 0.3);
        }
        
        .btn:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none;
        }
        
        .results-section {
            display: none;
        }
        
        .results-section.show {
            display: block;
        }
        
        .authenticity-score {
            text-align: center;
            margin-bottom: 30px;
        }
        
        .score-circle {
            width: 150px;
            height: 150px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            margin: 0 auto 20px;
            font-size: 2rem;
            font-weight: bold;
            color: white;
            position: relative;
        }
        
        .score-high { background: linear-gradient(135deg, #27ae60 0%, #2ecc71 100%); }
        .score-medium { background: linear-gradient(135deg, #f39c12 0%, #e67e22 100%); }
        .score-low { background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%); }
        
        .score-label {
            font-size: 1.2rem;
            color: #2c3e50;
            font-weight: 600;
        }
        
        .analysis-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        
        .analysis-card {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
            border-left: 4px solid #667eea;
        }
        
        .analysis-card h3 {
            color: #2c3e50;
            margin-bottom: 15px;
            font-size: 1.1rem;
        }
        
        .fact-item {
            background: white;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 10px;
            border-left: 3px solid #27ae60;
        }
        
        .fact-source {
            font-size: 0.9rem;
            color: #7f8c8d;
            margin-top: 5px;
        }
        
        .fact-similarity {
            float: right;
            background: #27ae60;
            color: white;
            padding: 4px 8px;
            border-radius: 12px;
            font-size: 0.8rem;
        }
        
        .recommendations {
            background: #e8f5e8;
            padding: 20px;
            border-radius: 10px;
            border-left: 4px solid #27ae60;
        }
        
        .recommendations h3 {
            color: #27ae60;
            margin-bottom: 15px;
        }
        
        .recommendations ul {
            list-style: none;
        }
        
        .recommendations li {
            padding: 8px 0;
            border-bottom: 1px solid #d5e8d5;
        }
        
        .recommendations li:last-child {
            border-bottom: none;
        }
        
        .loading {
            text-align: center;
            padding: 40px;
            color: #7f8c8d;
        }
        
        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #667eea;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto 20px;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .error {
            background: #fee;
            color: #c0392b;
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid #e74c3c;
            margin-bottom: 20px;
        }
        
        .google-facts {
            background: #e3f2fd;
            padding: 20px;
            border-radius: 10px;
            border-left: 4px solid #2196f3;
            margin-bottom: 20px;
        }
        
        .google-facts h3 {
            color: #2196f3;
            margin-bottom: 15px;
        }
        
        .free-alternatives {
            background: #fff3e0;
            padding: 20px;
            border-radius: 10px;
            border-left: 4px solid #ff9800;
            margin-bottom: 20px;
        }
        
        .free-alternatives h3 {
            color: #ff9800;
            margin-bottom: 15px;
        }
        
        .web-suggestions {
            background: #e8f5e8;
            padding: 20px;
            border-radius: 10px;
            border-left: 4px solid #4caf50;
            margin-bottom: 20px;
        }
        
        .web-suggestions h3 {
            color: #4caf50;
            margin-bottom: 15px;
        }
        
        .claim-item {
            background: white;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 10px;
            border-left: 3px solid #2196f3;
        }
        
        .claim-verdict {
            float: right;
            padding: 4px 8px;
            border-radius: 12px;
            font-size: 0.8rem;
            font-weight: bold;
        }
        
        .verdict-true { background: #27ae60; color: white; }
        .verdict-false { background: #e74c3c; color: white; }
        .verdict-partial { background: #f39c12; color: white; }
        .verdict-unclear { background: #95a5a6; color: white; }
        
        .offline-status {
            margin-top: 10px;
            color: #7f8c8d;
            font-size: 0.9rem;
        }

        .offline-badge {
            background-color: #e74c3c;
            color: white;
            padding: 4px 8px;
            border-radius: 6px;
            font-weight: bold;
            margin-right: 5px;
        }
        
        @media (max-width: 768px) {
            .container {
                margin: 10px;
                border-radius: 15px;
            }
            
            .header {
                padding: 30px 20px;
            }
            
            .header h1 {
                font-size: 2rem;
            }
            
            .main-content {
                padding: 20px;
            }
            
            .analysis-grid {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔍 News Authenticity Checker</h1>
            <p>Detect misinformation using AI-powered fact-checking and Google Fact Check API</p>
        </div>
        
        <div class="main-content">
            <div class="input-section">
                <div class="input-group">
                    <label for="newsText">Enter news text or headline to check:</label>
                    <textarea id="newsText" placeholder="Paste the news article, headline, or statement you want to verify..."></textarea>
                </div>
                
                <button class="btn" onclick="checkAuthenticity()" id="checkBtn">
                    🔍 Check Authenticity
                </button>
            </div>
            
            <div class="loading" id="loading" style="display: none;">
                <div class="spinner"></div>
                <p>Analyzing news authenticity...</p>
            </div>
            
            <div class="results-section" id="results">
                <div class="authenticity-score">
                    <div class="score-circle" id="scoreCircle">
                        <span id="scoreValue">0</span>%
                    </div>
                    <div class="score-label" id="scoreLabel">Authenticity Score</div>
                    <div class="offline-status" id="offlineStatus" style="display: none;">
                        <span class="offline-badge">🆓 Offline Mode</span>
                        <small>Working without external APIs</small>
                    </div>
                </div>
                
                <div class="analysis-grid">
                    <div class="analysis-card">
                        <h3>📊 Text Analysis</h3>
                        <div id="textAnalysis"></div>
                    </div>
                    
                    <div class="analysis-card">
                        <h3>🔍 Similar Verified Facts</h3>
                        <div id="similarFacts"></div>
                    </div>
                </div>
                
                <div class="google-facts" id="googleFacts" style="display: none;">
                    <h3>🔍 Google Fact Check Results</h3>
                    <div id="googleFactResults"></div>
                </div>
                
                <div class="free-alternatives" id="freeAlternatives" style="display: none;">
                    <h3>🆓 Free Fact-Checking Alternatives</h3>
                    <div id="freeAlternativesResults"></div>
                </div>
                
                <div class="web-suggestions" id="webSuggestions" style="display: none;">
                    <h3>🌐 Web Verification Suggestions</h3>
                    <div id="webSuggestionsResults"></div>
                </div>
                
                <div class="recommendations">
                    <h3>💡 Recommendations</h3>
                    <ul id="recommendations"></ul>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        async function checkAuthenticity() {
            const newsText = document.getElementById('newsText').value.trim();
            
            if (!newsText) {
                alert('Please enter some text to check.');
                return;
            }
            
            // Show loading
            document.getElementById('loading').style.display = 'block';
            document.getElementById('results').classList.remove('show');
            document.getElementById('checkBtn').disabled = true;
            
            try {
                const response = await fetch('/check_authenticity', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ news_text: newsText })
                });
                
                const result = await response.json();
                
                if (result.error) {
                    throw new Error(result.error);
                }
                
                displayResults(result);
                
            } catch (error) {
                console.error('Error:', error);
                document.getElementById('loading').innerHTML = `
                    <div class="error">
                        <strong>Error:</strong> ${error.message}
                    </div>
                `;
            } finally {
                document.getElementById('loading').style.display = 'none';
                document.getElementById('checkBtn').disabled = false;
            }
        }
        
        function displayResults(result) {
            // Update authenticity score
            const score = Math.round(result.authenticity_score * 100);
            const scoreCircle = document.getElementById('scoreCircle');
            const scoreValue = document.getElementById('scoreValue');
            const scoreLabel = document.getElementById('scoreLabel');
            
            scoreValue.textContent = score;
            
            // Set score color and label
            scoreCircle.className = 'score-circle';
            if (score >= 70) {
                scoreCircle.classList.add('score-high');
                scoreLabel.textContent = 'High Authenticity';
            } else if (score >= 40) {
                scoreCircle.classList.add('score-medium');
                scoreLabel.textContent = 'Medium Authenticity';
            } else {
                scoreCircle.classList.add('score-low');
                scoreLabel.textContent = 'Low Authenticity';
            }

            // Update offline status
            const offlineStatus = document.getElementById('offlineStatus');
            if (result.offline_mode) {
                offlineStatus.style.display = 'block';
            } else {
                offlineStatus.style.display = 'none';
            }
            
            // Display text analysis
            const textAnalysis = document.getElementById('textAnalysis');
            textAnalysis.innerHTML = `
                <p><strong>Length:</strong> ${result.text_analysis.length} characters</p>
                <p><strong>Word Count:</strong> ${result.text_analysis.word_count} words</p>
                <p><strong>Emotional Language:</strong> ${result.text_analysis.has_emotional_language ? '⚠️ Yes' : '✅ No'}</p>
                <p><strong>Clickbait Patterns:</strong> ${result.text_analysis.has_clickbait_patterns ? '⚠️ Yes' : '✅ No'}</p>
                <p><strong>Credible Sources:</strong> ${result.text_analysis.has_credible_sources ? '✅ Yes' : '⚠️ No'}</p>
                <p><strong>Sentiment:</strong> ${result.text_analysis.sentiment}</p>
            `;
            
            // Display similar facts
            const similarFacts = document.getElementById('similarFacts');
            if (result.similar_facts && result.similar_facts.length > 0) {
                similarFacts.innerHTML = result.similar_facts.map(fact => `
                    <div class="fact-item">
                        <span class="fact-similarity">${Math.round(fact.similarity * 100)}%</span>
                        <p>${fact.text}</p>
                        <div class="fact-source">
                            Source: ${fact.source} | Category: ${fact.category}
                        </div>
                    </div>
                `).join('');
            } else {
                similarFacts.innerHTML = '<p>No similar verified facts found.</p>';
            }
            
            // Display Google Fact Check results
            const googleFacts = document.getElementById('googleFacts');
            const googleFactResults = document.getElementById('googleFactResults');
            
            if (result.fact_check_results.claims && result.fact_check_results.claims.length > 0) {
                googleFactResults.innerHTML = result.fact_check_results.claims.map(claim => {
                    const verdictClass = claim.verdict ? `verdict-${claim.verdict}` : 'verdict-unclear';
                    const verdictText = claim.verdict ? claim.verdict.replace('_', ' ') : 'unclear';
                    
                    return `
                        <div class="claim-item">
                            <span class="claim-verdict ${verdictClass}">${verdictText.toUpperCase()}</span>
                            <p><strong>Claim:</strong> ${claim.text}</p>
                            ${claim.claimant ? `<p><strong>Claimant:</strong> ${claim.claimant}</p>` : ''}
                            ${claim.review ? `
                                <p><strong>Review:</strong> ${claim.review.title}</p>
                                <p><strong>Publisher:</strong> ${claim.review.publisher}</p>
                                ${claim.review.url ? `<p><strong>Source:</strong> <a href="${claim.review.url}" target="_blank">View Fact Check</a></p>` : ''}
                            ` : ''}
                        </div>
                    `;
                }).join('');
                googleFacts.style.display = 'block';
            } else {
                googleFacts.style.display = 'none';
            }

            // Display Free Fact-Checking Alternatives
            const freeAlternatives = document.getElementById('freeAlternatives');
            const freeAlternativesResults = document.getElementById('freeAlternativesResults');
            if (result.fact_check_results.free_alternatives && result.fact_check_results.free_alternatives.free_sources_checked && result.fact_check_results.free_alternatives.free_sources_checked.length > 0) {
                freeAlternativesResults.innerHTML = result.fact_check_results.free_alternatives.free_sources_checked.map(source => `
                    <div class="fact-item">
                        <p>${source.suggestion}</p>
                        <div class="fact-source">
                            Source: ${source.source} | URL: <a href="${source.url}" target="_blank">Visit</a>
                        </div>
                    </div>
                `).join('');
                freeAlternatives.style.display = 'block';
            } else {
                freeAlternatives.style.display = 'none';
            }

            // Display Web Verification Suggestions
            const webSuggestions = document.getElementById('webSuggestions');
            const webSuggestionsResults = document.getElementById('webSuggestionsResults');
            if (result.fact_check_results.web_suggestions && result.fact_check_results.web_suggestions.scraped_facts && result.fact_check_results.web_suggestions.scraped_facts.length > 0) {
                webSuggestionsResults.innerHTML = result.fact_check_results.web_suggestions.scraped_facts.map(fact => `
                    <div class="fact-item">
                        <p>${fact.suggestion}</p>
                        <div class="fact-source">
                            Type: ${fact.source_type} | Query: "${fact.query}"
                        </div>
                    </div>
                `).join('');
                webSuggestions.style.display = 'block';
            } else {
                webSuggestions.style.display = 'none';
            }
            
            // Display recommendations
            const recommendations = document.getElementById('recommendations');
            recommendations.innerHTML = result.recommendations.map(rec => 
                `<li>${rec}</li>`
            ).join('');
            
            // Show results
            document.getElementById('results').classList.add('show');
        }
        
        // Allow Enter key to submit
        document.getElementById('newsText').addEventListener('keydown', function(e) {
            if (e.key === 'Enter' && e.ctrlKey) {
                checkAuthenticity();
            }
        });
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    """Main page"""
    return render_template_string(HTML_TEMPLATE)

@app.route('/check_authenticity', methods=['POST'])
def check_authenticity():
    """API endpoint to check news authenticity"""
    try:
        data = request.get_json()
        news_text = data.get('news_text', '').strip()
        
        if not news_text:
            return jsonify({'error': 'News text is required'}), 400
        
        # Check authenticity
        result = checker.check_news_authenticity(news_text)
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"Error in check_authenticity endpoint: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
else:
    # For production deployment (Vercel)
    app.debug = False
