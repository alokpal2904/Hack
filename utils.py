import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from huggingface_hub import InferenceClient
import requests
import os
import warnings
warnings.filterwarnings('ignore')

def load_data(file):
    """
    Load CSV data into a pandas DataFrame
    
    Args:
        file: Uploaded file object from Streamlit
        
    Returns:
        pandas.DataFrame: Loaded data or None if error
    """
    try:
        df = pd.read_csv(file)
        
        # Validate required columns
        required_columns = ['user_id', 'step', 'completed']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            print(f"Missing required columns: {missing_columns}")
            return None
        
        # Convert completed column to boolean if it's numeric
        if df['completed'].dtype in ['int64', 'float64']:
            df['completed'] = df['completed'].astype(bool)
        
        # Ensure timestamp is datetime if present
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        
        return df
        
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return None

def analyze_funnel(df):
    """
    Analyze onboarding funnel and calculate drop-off rates
    
    Args:
        df: pandas.DataFrame with user_id, step, completed columns
        
    Returns:
        dict: Funnel analysis results with steps, user counts, drop-off rates
    """
    try:
        # Get unique steps in order
        steps = df['step'].unique()
        
        # Calculate users at each step
        step_counts = []
        for step in steps:
            users_at_step = df[df['step'] == step]['user_id'].nunique()
            step_counts.append(users_at_step)
        
        # Calculate drop-off rates
        dropoff_rates = []
        retention_rates = []
        
        for i, count in enumerate(step_counts):
            if i == 0:
                dropoff_rate = 0
                retention_rate = 100
            else:
                dropoff_rate = ((step_counts[i-1] - count) / step_counts[i-1]) * 100
                retention_rate = (count / step_counts[i-1]) * 100
            
            dropoff_rates.append(dropoff_rate)
            retention_rates.append(retention_rate)
        
        return {
            'step': steps.tolist(),
            'users': step_counts,
            'dropoff_rate': dropoff_rates,
            'retention_rate': retention_rates
        }
        
    except Exception as e:
        print(f"Error analyzing funnel: {str(e)}")
        return None

def cluster_dropoffs(df):
    """
    Cluster users who dropped off at each step using K-means
    
    Args:
        df: pandas.DataFrame with user data
        
    Returns:
        dict: Clustering results for each step
    """
    try:
        results = {}
        
        # Get unique steps
        steps = df['step'].unique()
        
        for step in steps:
            # Get users who dropped off at this step
            step_data = df[df['step'] == step]
            dropoff_users = step_data[~step_data['completed']]['user_id'].unique()
            
            if len(dropoff_users) < 3:  # Need at least 3 users for clustering
                results[step] = {
                    'total_dropoffs': len(dropoff_users),
                    'clusters': [],
                    'message': 'Insufficient data for clustering'
                }
                continue
            
            # Get user properties for clustering
            user_properties = []
            user_ids = []
            
            for user_id in dropoff_users:
                try:
                    user_data = df[df['user_id'] == user_id]
                    
                    # Extract features
                    features = {}
                    
                    # Email domain (categorical)
                    if 'email_domain' in df.columns:
                        try:
                            domain = user_data['email_domain'].iloc[0] if not user_data['email_domain'].empty else 'unknown'
                            features['email_domain'] = domain
                        except (IndexError, KeyError):
                            features['email_domain'] = 'unknown'
                    
                    # Device (categorical)
                    if 'device' in df.columns:
                        try:
                            device = user_data['device'].iloc[0] if not user_data['device'].empty else 'unknown'
                            features['device'] = device
                        except (IndexError, KeyError):
                            features['device'] = 'unknown'
                    
                    # Location (categorical)
                    if 'location' in df.columns:
                        try:
                            location = user_data['location'].iloc[0] if not user_data['location'].empty else 'unknown'
                            features['location'] = location
                        except (IndexError, KeyError):
                            features['location'] = 'unknown'
                    
                    # Time-based features
                    if 'timestamp' in df.columns:
                        try:
                            timestamp = user_data['timestamp'].iloc[0] if not user_data['timestamp'].empty else pd.Timestamp.now()
                            if pd.notna(timestamp):
                                features['hour'] = timestamp.hour
                                features['day_of_week'] = timestamp.dayofweek
                                features['is_weekend'] = 1 if timestamp.dayofweek >= 5 else 0
                        except (IndexError, KeyError, AttributeError):
                            # Use default values if timestamp processing fails
                            features['hour'] = 12
                            features['day_of_week'] = 0
                            features['is_weekend'] = 0
                    
                    user_properties.append(features)
                    user_ids.append(user_id)
                except Exception as user_error:
                    print(f"Error processing user {user_id}: {str(user_error)}")
                    continue
            
            if not user_properties:
                results[step] = {
                    'total_dropoffs': len(dropoff_users),
                    'clusters': [],
                    'message': 'No properties available for clustering'
                }
                continue
            
            # Convert to DataFrame for processing
            properties_df = pd.DataFrame(user_properties)
            
            # Encode categorical variables
            encoded_features = []
            feature_names = []
            
            for col in properties_df.columns:
                if properties_df[col].dtype == 'object':
                    # Categorical variable
                    le = LabelEncoder()
                    encoded = le.fit_transform(properties_df[col].fillna('unknown'))
                    encoded_features.append(encoded)
                    feature_names.append(f"{col}_encoded")
                else:
                    # Numerical variable
                    encoded_features.append(properties_df[col].fillna(0).values)
                    feature_names.append(col)
            
            if not encoded_features:
                results[step] = {
                    'total_dropoffs': len(dropoff_users),
                    'clusters': [],
                    'message': 'No features available for clustering'
                }
                continue
            
            # Combine features
            X = np.column_stack(encoded_features)
            
            # Standardize features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Determine optimal number of clusters (max 3 for interpretability)
            max_clusters = min(3, len(dropoff_users) - 1)
            if max_clusters < 2:
                max_clusters = 2
            
            # Perform clustering
            try:
                kmeans = KMeans(n_clusters=max_clusters, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(X_scaled)
                
                # Analyze clusters
                clusters_info = []
                for cluster_id in range(max_clusters):
                    try:
                        cluster_mask = cluster_labels == cluster_id
                        cluster_size = np.sum(cluster_mask)
                        
                        if cluster_size == 0 or cluster_size < 1:
                            continue
                        
                        # Ensure cluster_mask is valid
                        if not isinstance(cluster_mask, np.ndarray) or len(cluster_mask) == 0:
                            continue
                        
                        # Get characteristics of this cluster
                        cluster_features = properties_df.iloc[cluster_mask]
                        characteristics = []
                        
                        for col in properties_df.columns:
                            if col in ['email_domain', 'device', 'location']:
                                try:
                                    if len(cluster_features[col]) > 0:
                                        most_common = cluster_features[col].mode()
                                        if len(most_common) > 0:
                                            characteristics.append(f"{col}: {most_common.iloc[0]}")
                                        else:
                                            characteristics.append(f"{col}: N/A")
                                    else:
                                        characteristics.append(f"{col}: N/A")
                                except (IndexError, KeyError, AttributeError):
                                    characteristics.append(f"{col}: N/A")
                            elif col in ['hour', 'day_of_week']:
                                try:
                                    if len(cluster_features[col]) > 0:
                                        avg_val = cluster_features[col].mean()
                                        if pd.notna(avg_val) and not pd.isna(avg_val):
                                            if col == 'hour':
                                                characteristics.append(f"avg_hour: {avg_val:.1f}")
                                            elif col == 'day_of_week':
                                                day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
                                                day_idx = int(avg_val)
                                                if 0 <= day_idx < 7:
                                                    characteristics.append(f"avg_day: {day_names[day_idx]}")
                                                else:
                                                    characteristics.append(f"avg_day: N/A")
                                        else:
                                            characteristics.append(f"{col}: N/A")
                                    else:
                                        characteristics.append(f"{col}: N/A")
                                except (ValueError, TypeError, AttributeError):
                                    characteristics.append(f"{col}: N/A")
                            elif col == 'is_weekend':
                                try:
                                    if len(cluster_features[col]) > 0:
                                        weekend_ratio = cluster_features[col].mean()
                                        if pd.notna(weekend_ratio) and not pd.isna(weekend_ratio):
                                            characteristics.append(f"weekend_ratio: {weekend_ratio:.1f}")
                                        else:
                                            characteristics.append(f"{col}: N/A")
                                    else:
                                        characteristics.append(f"{col}: N/A")
                                except (ValueError, TypeError, AttributeError):
                                    characteristics.append(f"{col}: N/A")
                        
                        # Safely get user IDs for this cluster
                        try:
                            cluster_user_ids = [user_ids[i] for i in range(len(user_ids)) if cluster_mask[i]]
                        except (IndexError, TypeError):
                            cluster_user_ids = []
                        
                        clusters_info.append({
                            'size': int(cluster_size),
                            'characteristics': '; '.join(characteristics),
                            'user_ids': cluster_user_ids
                        })
                        
                    except Exception as cluster_error:
                        print(f"Error analyzing cluster {cluster_id}: {str(cluster_error)}")
                        continue
                
                results[step] = {
                    'total_dropoffs': len(dropoff_users),
                    'clusters': clusters_info
                }
            except Exception as clustering_error:
                print(f"Error in clustering algorithm for step {step}: {str(clustering_error)}")
                results[step] = {
                    'total_dropoffs': len(dropoff_users),
                    'clusters': [],
                    'message': f'Clustering failed: {str(clustering_error)}'
                }
        
        return results
        
    except Exception as e:
        print(f"Error clustering dropoffs: {str(e)}")
        return None

def query_llama(prompt: str) -> str:
    """
    Query models via OpenRouter API
    """
    try:
        openrouter_key = os.getenv("OPENROUTER_API_KEY")
        if not openrouter_key:
            print("OPENROUTER_API_KEY not found")
            return generate_fallback_insights(prompt)
        
        headers = {
            "Authorization": f"Bearer {openrouter_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "openai/gpt-3.5-turbo",  # Free tier available
            "messages": [
                {
                    "role": "system",
                    "content": "You are a data analyst and growth strategist. Analyze onboarding funnel data and provide actionable insights, hypotheses, and recommendations."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": 500,
            "temperature": 0.7
        }
        
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content'].strip()
        else:
            print(f"OpenRouter error: {response.status_code} - {response.text}")
            return generate_fallback_insights(prompt)
            
    except Exception as e:
        print(f"OpenRouter error: {str(e)}")
        return generate_fallback_insights(prompt)





def generate_fallback_insights(prompt):
    """
    Generate fallback insights when AI models are unavailable
    
    Args:
        prompt: str, the original prompt
        
    Returns:
        str: Generated insights based on common UX patterns
    """
    try:
        # Extract funnel information from prompt
        if "Funnel Steps:" in prompt and "Drop-off Rates:" in prompt:
            # Parse the prompt to extract key information
            lines = prompt.split('\n')
            steps = []
            rates = []
            
            for line in lines:
                if "Funnel Steps:" in line:
                    # Extract steps (this is a simplified parser)
                    steps_text = line.split("Funnel Steps:")[1].strip()
                    if "[" in steps_text and "]" in steps_text:
                        steps = eval(steps_text)  # Safe for our controlled input
                elif "Drop-off Rates:" in line:
                    # Extract rates
                    rates_text = line.split("Drop-off Rates:")[1].strip()
                    if "[" in rates_text and "]" in rates_text:
                        rates = eval(rates_text)  # Safe for our controlled input
            
            if steps and rates:
                return generate_structured_insights(steps, rates)
        
        # Generic fallback if we can't parse the prompt
        return """🤖 **AI Insights (Fallback Mode)**

Since the AI model is currently unavailable, here are general UX improvement recommendations for onboarding funnels:

**📊 Common Drop-off Patterns:**
- **First Step Drop-offs**: Usually indicate unclear value proposition or technical issues
- **Middle Step Drop-offs**: Often suggest complexity or friction in the process
- **Final Step Drop-offs**: May indicate last-minute concerns or form fatigue

**🔧 General UX Recommendations:**
1. **Simplify the Process**: Reduce the number of steps and required fields
2. **Progress Indicators**: Show users where they are in the onboarding flow
3. **Mobile Optimization**: Ensure mobile users can complete all steps easily
4. **Clear Value Proposition**: Make the benefits obvious at each step
5. **Error Handling**: Provide helpful error messages and recovery options

**🧪 A/B Testing Ideas:**
- Test different step orders
- Compare single-page vs. multi-step forms
- Experiment with progress indicators
- Test different copy and messaging

**🎯 Priority Actions:**
1. Identify the step with the highest drop-off rate
2. Analyze user behavior at that specific step
3. Implement one improvement and measure the impact
4. Iterate based on results

*Note: These are general recommendations. For specific insights, please ensure your API key is configured correctly.*"""
        
    except Exception as e:
        print(f"Error generating fallback insights: {str(e)}")
        return "AI insights could not be generated. Please check your API configuration and try again."

def generate_structured_insights(steps, rates):
    """
    Generate structured insights based on funnel data
    
    Args:
        steps: list of funnel steps
        rates: list of drop-off rates
        
    Returns:
        str: Structured insights
    """
    try:
        insights = "🤖 **AI-Generated Insights (Fallback Mode)**\n\n"
        
        # Find the step with highest drop-off
        if rates and len(rates) > 1:
            max_dropoff_idx = rates.index(max(rates[1:]))  # Skip first step (0% dropoff)
            max_dropoff_step = steps[max_dropoff_idx] if max_dropoff_idx < len(steps) else "unknown"
            max_dropoff_rate = rates[max_dropoff_idx]
            
            insights += f"**🚨 Critical Issue Identified:**\n"
            insights += f"The highest drop-off occurs at **{max_dropoff_step}** with a **{max_dropoff_rate:.1f}%** drop-off rate.\n\n"
        
        insights += "**📊 Step-by-Step Analysis:**\n"
        for i, (step, rate) in enumerate(zip(steps, rates)):
            if i == 0:
                insights += f"• **{step}**: Starting point (0% drop-off)\n"
            else:
                insights += f"• **{step}**: {rate:.1f}% drop-off from previous step\n"
        
        insights += "\n**🔧 Specific Recommendations:**\n"
        
        # Generate recommendations based on step names
        for step in steps:
            step_lower = step.lower()
            if "signup" in step_lower or "register" in step_lower:
                insights += f"• **{step}**: Simplify form fields, add social login options\n"
            elif "verification" in step_lower:
                insights += f"• **{step}**: Ensure email delivery, add SMS backup\n"
            elif "profile" in step_lower:
                insights += f"• **{step}**: Make optional fields truly optional, add progress bar\n"
            elif "preferences" in step_lower:
                insights += f"• **{step}**: Use smart defaults, reduce choice overload\n"
            else:
                insights += f"• **{step}**: Review user flow, add help text and examples\n"
        
        insights += "\n**🧪 A/B Testing Priority:**\n"
        insights += "1. Test the step with highest drop-off first\n"
        insights += "2. Compare different UI layouts and copy\n"
        insights += "3. Test mobile vs. desktop experiences\n"
        insights += "4. Experiment with step order and grouping\n"
        
        insights += "\n**🎯 Immediate Actions:**\n"
        insights += "1. Review the user experience at the highest drop-off step\n"
        insights += "2. Check for technical issues or form validation problems\n"
        insights += "3. Implement one improvement and measure impact\n"
        insights += "4. Gather user feedback through surveys or interviews\n"
        
        insights += "\n*Note: These insights are generated using fallback analysis. For AI-powered recommendations, please configure your API key.*"
        
        return insights
        
    except Exception as e:
        print(f"Error generating structured insights: {str(e)}")
        return "Could not generate structured insights. Please check your data format."

def create_sample_data():
    """
    Create sample data for testing purposes
    
    Returns:
        pandas.DataFrame: Sample onboarding data
    """
    np.random.seed(42)
    
    # Generate sample data
    n_users = 1000
    steps = ['signup', 'email_verification', 'profile_setup', 'preferences', 'onboarding_complete']
    
    data = []
    
    for user_id in range(1, n_users + 1):
        # Random user properties
        email_domains = ['gmail.com', 'yahoo.com', 'outlook.com', 'hotmail.com', 'icloud.com']
        devices = ['mobile', 'desktop', 'tablet']
        locations = ['US', 'UK', 'CA', 'AU', 'DE', 'FR', 'JP', 'IN']
        
        email_domain = np.random.choice(email_domains)
        device = np.random.choice(devices)
        location = np.random.choice(locations)
        
        # Simulate funnel progression with realistic drop-offs
        current_step = 0
        for step in steps:
            # Probability of completing each step (decreases as we go deeper)
            completion_prob = 0.95 - (current_step * 0.15)
            completed = np.random.random() < completion_prob
            
            # Add some correlation between user properties and completion
            if device == 'mobile' and step in ['profile_setup', 'preferences']:
                completion_prob *= 0.8  # Mobile users struggle more with complex steps
            
            if email_domain in ['gmail.com', 'outlook.com'] and step == 'email_verification':
                completion_prob *= 1.1  # Popular domains have higher verification success
            
            completed = np.random.random() < completion_prob
            
            # Add timestamp
            base_time = pd.Timestamp('2024-01-01 09:00:00')
            time_offset = pd.Timedelta(hours=np.random.randint(0, 24*30))  # Random time within a month
            timestamp = base_time + time_offset
            
            data.append({
                'user_id': f'user_{user_id:03d}',
                'step': step,
                'completed': completed,
                'email_domain': email_domain,
                'device': device,
                'location': location,
                'timestamp': timestamp
            })
            
            if not completed:
                break  # User dropped off at this step
            
            current_step += 1
    
    return pd.DataFrame(data)






# import os
# from huggingface_hub import InferenceClient
# from dotenv import load_dotenv 
# # Load .env file 
# load_dotenv() 
# HF_API_KEY = os.getenv("HF_API_KEY") 
# if HF_API_KEY is None:
#     raise ValueError("HF_API_KEY is not set. Please check your .env file.")
# MODEL_ID = "meta-llama/Llama-3.1-8B" 
# client = InferenceClient(model=MODEL_ID, token=HF_API_KEY)
# def query_llm(prompt: str) -> str:
    
#     try: 
#         response = client.text_generation( prompt, max_new_tokens=256, temperature=0.7, do_sample=True ) 
#         return response
#     except Exception as e:
#         return f"Error in query_llm: {e}"







