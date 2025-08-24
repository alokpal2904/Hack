# 🚀 Onboarding Drop-off Analyzer

A powerful Streamlit application that analyzes user onboarding funnels to identify drop-off points, cluster users who drop off, and generate AI-powered insights for UX improvements.

## ✨ Features

- **📊 Funnel Analysis**: Visualize user progression through onboarding steps with drop-off percentages
- **🔍 User Clustering**: Identify patterns in users who drop off using K-means clustering
- **🤖 AI Insights**: Get intelligent hypotheses and UX recommendations using Llama 3.1
- **📈 Interactive Charts**: Beautiful Plotly visualizations for data exploration
- **💾 Data Export**: Download analysis results and processed data
- **📱 Responsive UI**: Modern Streamlit interface with sidebar navigation

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- Hugging Face API key (for AI insights)

### Setup Steps

1. **Clone or download the project files**
   ```bash
   # If you have git installed
   git clone <repository-url>
   cd onboarding-dropoff-analyzer
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   Create a `.env` file in the project root:
   ```bash
   # .env
   HF_API_KEY=your_huggingface_token_here
   ```
   
   **To get a Hugging Face API key:**
   - Go to [Hugging Face](https://huggingface.co/)
   - Sign up/login and go to Settings → Access Tokens
   - Create a new token with read permissions

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

5. **Open your browser**
   Navigate to `http://localhost:8501`

## 📊 Expected Data Format

Your CSV file should contain the following columns:

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `user_id` | String | Unique user identifier | `user_001`, `john_doe` |
| `step` | String | Onboarding step name | `signup`, `profile_setup` |
| `completed` | Boolean/Integer | Whether step was completed | `True/False` or `1/0` |
| `email_domain` | String | User's email domain | `gmail.com`, `yahoo.com` |
| `device` | String | User's device type | `mobile`, `desktop`, `tablet` |
| `location` | String | User's location | `US`, `UK`, `CA` |
| `timestamp` | DateTime | When step was attempted | `2024-01-01 10:00:00` |

### Sample Data Structure
```csv
user_id,step,completed,email_domain,device,location,timestamp
user_001,signup,True,gmail.com,mobile,US,2024-01-01 10:00:00
user_001,profile_setup,False,gmail.com,mobile,US,2024-01-01 10:05:00
user_002,signup,True,yahoo.com,desktop,UK,2024-01-01 11:00:00
user_002,profile_setup,True,yahoo.com,desktop,UK,2024-01-01 11:05:00
```

## 🎯 How It Works

### 1. Data Upload & Processing
- Upload your CSV file through the Streamlit interface
- Data is automatically validated and processed
- Missing or invalid data is handled gracefully

### 2. Funnel Analysis
- Calculates user counts at each onboarding step
- Computes drop-off and retention rates
- Generates interactive funnel charts using Plotly

### 3. User Clustering
- Identifies users who dropped off at each step
- Uses K-means clustering to group similar users
- Analyzes patterns in email domains, devices, locations, and timing

### 4. AI-Powered Insights
- Sends funnel data to Llama 3.1 model via Hugging Face
- Generates hypotheses about why users drop off
- Provides actionable UX recommendations
- Suggests A/B testing ideas

### 5. Results & Export
- View all analysis results in the web interface
- Download funnel summaries and processed data
- Share insights with your team

## 🔧 Configuration

### Environment Variables
- `HF_API_KEY`: Your Hugging Face API token for AI insights

### Customization Options
You can modify the following in `utils.py`:
- Clustering parameters (number of clusters, features used)
- LLM prompt engineering
- Data preprocessing logic

## 📁 Project Structure

```
onboarding-dropoff-analyzer/
├── app.py              # Main Streamlit application
├── utils.py            # Helper functions and analysis logic
├── requirements.txt    # Python dependencies
├── README.md          # This file
└── .env               # Environment variables (create this)
```

## 🚀 Usage Examples

### Basic Analysis
1. Upload your CSV file
2. View the funnel chart and drop-off statistics
3. Explore user clusters for each step
4. Generate AI insights for improvement ideas

### Advanced Analysis
- Compare different time periods
- Analyze specific user segments
- Export results for further analysis
- Use insights to prioritize UX improvements

## 🐛 Troubleshooting

### Common Issues

**"HF_API_KEY not found"**
- Ensure you've created a `.env` file
- Check that your API key is correct
- Verify the file is in the project root

**"No data found in uploaded file"**
- Check your CSV format matches the expected structure
- Ensure required columns are present
- Verify the file isn't empty

**Clustering fails**
- Ensure you have at least 3 users who dropped off at each step
- Check that user properties (email, device, location) are populated

**LLM insights fail**
- Verify your Hugging Face API key is valid
- Check your internet connection
- Ensure the model is available (may require specific access)

## 🤝 Contributing

Feel free to contribute improvements:
- Bug fixes
- New features
- Documentation updates
- Performance optimizations

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- Built with [Streamlit](https://streamlit.io/) for the web interface
- Data visualization powered by [Plotly](https://plotly.com/)
- Machine learning with [scikit-learn](https://scikit-learn.org/)
- AI insights via [Hugging Face](https://huggingface.co/) and Llama 3.1

---

**Happy analyzing! 🎉**

