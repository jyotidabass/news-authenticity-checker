# 🔍 News Authenticity Checker

A comprehensive fake news detection tool that works **completely offline by default** with **optional API enhancements** for better accuracy. Built with Flask, AI text analysis, and multiple external fact-checking APIs.

## ✨ Key Features

### 🆓 **Always Available (Offline)**
- **AI Text Analysis** - Detects suspicious patterns, emotional language, and clickbait
- **Local Fact Database** - Built-in verified facts for comparison
- **Similarity Matching** - Finds similar verified facts using embeddings
- **Authenticity Scoring** - Calculates trustworthiness score
- **Free Fact-Checking Sources** - Manual verification suggestions
- **Web Scraping Analysis** - Extracts facts from reliable sources

### 🚀 **Optional API Enhancements**
- **Google Fact Check API** - Real-time fact verification from Google's database
- **News API** - Related articles and current news context
- **OpenAI API** - Advanced AI-powered analysis and reasoning
- **Pinecone** - Enhanced vector similarity search
- **Dynamic Configuration** - Add/remove APIs through web interface
- **Real-time Status Monitoring** - See which APIs are active

## 🎯 **How It Works**

1. **Default Mode**: App runs completely offline with basic analysis
2. **Enhanced Mode**: Users can optionally add API keys for better results
3. **Mixed Mode**: Some APIs can be active while others are inactive
4. **Always Safe**: Even if APIs fail, users still get comprehensive results

## 🚀 **Quick Start**

### Option 1: Run Locally (Recommended for Development)

```bash
# Clone the repository
git clone <your-repo-url>
cd fake-news-detector

# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py

# Open http://localhost:5000 in your browser
```

### Option 2: Deploy to Vercel (Production)

```bash
# Install Vercel CLI
npm i -g vercel

# Deploy
vercel --prod
```

## 🔑 **API Configuration (Optional)**

### **Web-Based Configuration (Recommended)**
1. Open the app in your browser
2. Click "⚙️ API Configuration (Optional)"
3. Add your API keys in the form
4. Click "💾 Save API Configuration"
5. APIs are automatically tested and activated

### **Manual Configuration (Alternative)**
Create a `.env` file in your project root:

```env
# Optional APIs for enhanced results
GOOGLE_API_KEY=your_google_fact_check_api_key
NEWS_API_KEY=your_news_api_key
OPENAI_API_KEY=your_openai_api_key
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_ENVIRONMENT=your_pinecone_environment
```

## 📚 **Getting API Keys**

### **Google Fact Check API**
1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Enable the "Fact Check Tools API"
3. Create credentials (API Key)
4. Add to your configuration

### **News API**
1. Visit [NewsAPI.org](https://newsapi.org/)
2. Sign up for a free account
3. Get your API key
4. Add to your configuration

### **OpenAI API**
1. Go to [OpenAI Platform](https://platform.openai.com/)
2. Create an account and add billing
3. Generate an API key
4. Add to your configuration

### **Pinecone**
1. Visit [Pinecone.io](https://www.pinecone.io/)
2. Create a free account
3. Get your API key and environment
4. Add to your configuration

## 🧪 **Testing**

### **Run the Test Suite**
```bash
python test_script.py
```

### **Test Individual Endpoints**
```bash
# Health check
curl http://localhost:5000/health

# API status
curl http://localhost:5000/api/status

# API configuration
curl http://localhost:5000/api/config

# Check authenticity
curl -X POST http://localhost:5000/check_authenticity \
  -H "Content-Type: application/json" \
  -d '{"news_text": "Test news article"}'
```

## 🏗️ **Architecture**

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Frontend      │    │   Flask Backend  │    │   External      │
│   (HTML/JS)     │◄──►│   (Python)       │◄──►│   APIs          │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   User Input    │    │   Text Analysis  │    │   Google Fact   │
│   Results       │    │   Embeddings     │    │   Check API     │
│   API Config    │    │   Similarity     │    │   News API      │
└─────────────────┘    │   Scoring        │    │   OpenAI API    │
                       └──────────────────┘    └─────────────────┘
```

## 📊 **API Status Monitoring**

The app provides real-time monitoring of all configured APIs:

- **🟢 Active**: API is working and enhancing results
- **🔴 Inactive**: API is not configured or failed
- **📊 Enhancement Level**: Percentage of active APIs
- **🔍 Real-time Testing**: APIs are tested before each use

## 🎨 **Customization**

### **Adding New APIs**
1. Add API key to `Config` class
2. Implement API method in `NewsAuthenticityChecker`
3. Add status checking in `check_api_status()`
4. Integrate in `check_news_authenticity()`
5. Update frontend display

### **Modifying Analysis**
- Adjust similarity thresholds in `Config.SIMILARITY_THRESHOLD`
- Modify text analysis patterns in `analyze_text_characteristics()`
- Update fact database in `Config.ENHANCED_FACTS`

## 🔒 **Security Features**

- API keys are masked in responses
- No sensitive data is logged
- Graceful error handling for API failures
- Secure configuration updates

## 📈 **Performance**

- **Offline Mode**: Instant results (no API calls)
- **API Mode**: Results enhanced with external data
- **Caching**: Embeddings and analysis are cached
- **Rate Limiting**: Respects API rate limits

## 🚨 **Troubleshooting**

### **APIs Not Working**
1. Check API status in the web interface
2. Verify API keys are correct
3. Ensure APIs are enabled in their respective platforms
4. Check network connectivity

### **Offline Mode Issues**
1. Verify all required Python packages are installed
2. Check if the model files are downloaded
3. Ensure sufficient disk space for embeddings

### **Deployment Issues**
1. Verify `vercel.json` configuration
2. Check environment variables in Vercel dashboard
3. Ensure Python version compatibility

## 🤝 **Contributing**

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📄 **License**

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 **Acknowledgments**

- **HuggingFace** for sentence transformers
- **Google** for Fact Check Tools API
- **OpenAI** for AI analysis capabilities
- **NewsAPI** for news aggregation
- **Pinecone** for vector similarity search

---

**💡 Remember**: This tool works perfectly without any APIs! APIs only enhance the results for better accuracy and real-time data.
