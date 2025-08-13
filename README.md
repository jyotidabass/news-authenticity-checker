# 🔍 News Authenticity Checker

A powerful AI-powered application that detects misinformation in news articles using advanced NLP techniques, **completely offline** with optional API enhancements.

**A comprehensive solution for detecting misinformation and verifying news authenticity**

## ✨ Features

- **🆓 100% Offline Capable**: Works without internet connection or API keys
- **AI-Powered Analysis**: Uses HuggingFace sentence transformers for semantic understanding
- **Enhanced Local Database**: Built-in verified facts covering health, science, technology, politics, and more
- **Text Characteristic Analysis**: Detects emotional language, clickbait patterns, and credibility indicators
- **Real-time Scoring**: Provides authenticity scores with detailed breakdowns
- **Beautiful UI**: Modern, responsive web interface with real-time feedback
- **Smart Recommendations**: AI-generated suggestions for fact verification
- **Multiple Verification Methods**: Combines offline analysis with optional API enhancements
- **Optional API Integration**: Google Fact Check, News API, and Pinecone for enhanced features

## 🚀 Technologies Used

- **Backend**: Flask (Python)
- **AI/ML**: HuggingFace Transformers, Sentence Transformers
- **Local Database**: Built-in verified facts (always available)
- **Optional**: Google Fact Check API, News API, Pinecone
- **Frontend**: HTML5, CSS3, JavaScript (Vanilla)
- **Deployment**: Vercel, GitHub, or any platform

## 🛠️ Installation & Setup

### Prerequisites

- Python 3.8+
- **No internet connection required** for core functionality
- **No API keys required** for basic fact-checking
- **Optional**: Google Cloud Platform account (for enhanced Fact Check API)
- **Optional**: News API account (free tier available)
- **Optional**: Pinecone account (for enhanced vector search)

> **🆓 100% Offline Ready!** The application works completely without internet connection or API keys. All core features are available offline with a built-in database of verified facts.

> **💡 Enhanced with APIs**: Optional API keys provide additional fact-checking capabilities and real-time data, but are not required for the app to function.

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/news-authenticity-checker.git
cd news-authenticity-checker
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Set Up Environment Variables (Optional)

> **💡 The app works without any API keys!** These are optional for enhanced functionality.

Create a `.env` file in the root directory:

```env
# Google Fact Check API (Optional - for enhanced fact-checking)
GOOGLE_API_KEY=your_google_api_key_here

# News API (Optional - free tier available)
NEWS_API_KEY=your_news_api_key_here

# Pinecone (Optional - for enhanced vector search)
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_ENVIRONMENT=your_pinecone_environment_here
```

**Free Alternatives Available:**
- **Fact-Checking Sources**: Snopes, FactCheck.org, PolitiFact, Reuters, BBC, AFP
- **Web Verification**: Search suggestions and manual verification steps
- **AI Analysis**: Text characteristics, sentiment analysis, credibility indicators
- **Local Fact Database**: Built-in verified facts for comparison

### 4. Get API Keys (Optional)

> **🆓 Skip this step if you want to use the app for free!** The application provides comprehensive fact-checking without any API keys.

#### Google Fact Check API (Optional - Enhanced Fact-Checking)
1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select existing one
3. Enable the Fact Check Tools API
4. Create credentials (API Key)
5. Copy the API key to your `.env` file

#### News API (Optional - Free Tier Available)
1. Sign up at [News API](https://newsapi.org/) (1000 requests/month free)
2. Get your API key
3. Add to `.env` file for related news articles

#### Pinecone (Optional - Enhanced Vector Search)
1. Sign up at [Pinecone](https://www.pinecone.io/)
2. Create a new index
3. Copy your API key and environment to `.env` file

**What You Get Without API Keys:**
- ✅ AI-powered text analysis
- ✅ Emotional language detection
- ✅ Clickbait pattern recognition
- ✅ Credibility source analysis
- ✅ Similar fact matching
- ✅ Free fact-checking source recommendations
- ✅ Manual verification guidance
- ✅ Authenticity scoring

### 5. Run the Application

```bash
python app.py
```

The application will be available at `http://localhost:5000`

## 🌐 Usage

1. **Open the Application**: Navigate to the web interface
2. **Input News Text**: Paste the news article, headline, or statement you want to verify
3. **Analyze**: Click "Check Authenticity" to start the analysis
4. **Review Results**: Get detailed breakdowns including:
   - Authenticity Score (0-100%)
   - Text Analysis (length, emotional language, clickbait patterns)
   - Similar Verified Facts
   - Google Fact Check Results
   - AI-Generated Recommendations

## 📊 How It Works

### 1. Text Processing
- Input text is processed and cleaned
- Key search terms are extracted for fact-checking
- Text characteristics are analyzed for credibility indicators

### 2. AI Analysis
- Sentence embeddings are generated using HuggingFace models
- Similarity scores are calculated against verified facts
- Text patterns are analyzed for fake news indicators

### 3. Fact Checking
- Google Fact Check API searches for related claims
- Verdicts are analyzed and categorized
- Confidence scores are calculated

### 4. Scoring Algorithm
The authenticity score is calculated based on:
- **Similar Facts (30%)**: How well the text matches verified information
- **Text Characteristics (40%)**: Language quality, source credibility, length
- **Fact Check Results (30%)**: Google API verification results

## 🌐 Offline vs Online Features

### 🆓 **Always Available (Offline)**
- ✅ **AI Text Analysis**: Emotional language, clickbait patterns, credibility indicators
- ✅ **Sentiment Analysis**: Positive, negative, or neutral content assessment
- ✅ **Local Fact Database**: 20+ verified facts covering multiple categories
- ✅ **Similarity Matching**: AI-powered fact comparison using embeddings
- ✅ **Authenticity Scoring**: Comprehensive scoring algorithm
- ✅ **Smart Recommendations**: AI-generated verification advice
- ✅ **Text Characteristics**: Length, word count, source credibility analysis

### 🌐 **Enhanced with APIs (Optional)**
- 🔍 **Google Fact Check**: Real-time fact verification from Google's database
- 📰 **News API**: Related articles and current news context
- 🗄️ **Pinecone**: Advanced vector similarity search
- 🔄 **Real-time Updates**: Live fact-checking data

### 📱 **Deployment Flexibility**
- **Local Development**: Works completely offline
- **Cloud Deployment**: Enhanced with optional APIs
- **Air-gapped Networks**: Full functionality without internet
- **Low-bandwidth Environments**: Core features work without external calls

## 🚀 Deployment

### Deploy to Vercel

1. **Push to GitHub**:
```bash
git add .
git commit -m "Initial commit"
git push origin main
```

2. **Connect to Vercel**:
   - Go to [Vercel](https://vercel.com/)
   - Import your GitHub repository
   - Set environment variables in Vercel dashboard
   - Deploy!

### Deploy to Other Platforms

The application is compatible with:
- **Heroku**: Add `Procfile` and `runtime.txt`
- **Railway**: Direct GitHub integration
- **DigitalOcean App Platform**: Container deployment
- **AWS/GCP**: Container or serverless deployment

## 🔧 Configuration

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `GOOGLE_API_KEY` | Google Fact Check API key | No |
| `PINECONE_API_KEY` | Pinecone API key | No |
| `PINECONE_ENVIRONMENT` | Pinecone environment | No |

### Customization

- **Model**: Change `MODEL_NAME` in `Config` class for different HuggingFace models
- **Thresholds**: Adjust `SIMILARITY_THRESHOLD` for sensitivity
- **Fact Database**: Modify `load_fact_database()` method for custom facts
- **UI**: Customize the HTML template and CSS styles

## 📈 Performance & Scaling

- **Caching**: Implement Redis for response caching
- **Rate Limiting**: Add rate limiting for API endpoints
- **Load Balancing**: Use multiple instances behind a load balancer
- **CDN**: Serve static assets through CDN for global performance

## 🧪 Testing

### Run Tests
```bash
python -m pytest tests/
```

### Test API Endpoints
```bash
# Health check
curl http://localhost:5000/health

# Check authenticity
curl -X POST http://localhost:5000/check_authenticity \
  -H "Content-Type: application/json" \
  -d '{"news_text": "Your news text here"}'
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [HuggingFace](https://huggingface.co/) for transformer models
- [Google Fact Check API](https://developers.google.com/fact-check/tools/api) for fact verification
- [Pinecone](https://www.pinecone.io/) for vector similarity search
- [Flask](https://flask.palletsprojects.com/) for the web framework

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/your-username/news-authenticity-checker/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-username/news-authenticity-checker/discussions)

## 🔮 Future Enhancements

- [ ] Multi-language support
- [ ] Real-time news monitoring
- [ ] Browser extension
- [ ] Mobile app
- [ ] Advanced sentiment analysis
- [ ] Social media integration
- [ ] Automated fact-checking reports
- [ ] Machine learning model training
- [ ] Integration with medical fact-checking databases
- [ ] Audio/video content analysis

---

**Built for a more informed world with AI-powered misinformation detection and fact verification systems.**
