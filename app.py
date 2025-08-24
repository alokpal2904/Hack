import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from utils import load_data, analyze_funnel, cluster_dropoffs, query_llama
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Onboarding Drop-off Analyzer",
    page_icon="📊",
    layout="wide"
)

# Sidebar
st.sidebar.title("📊 Onboarding Drop-off Analyzer")
st.sidebar.markdown("""
This tool analyzes user onboarding funnels to identify:
- **Drop-off points** at each step
- **User clusters** who are dropping off
- **AI-generated insights** for UX improvements

Upload a CSV file with columns:
- `user_id`: Unique user identifier
- `step`: Onboarding step name
- `completed`: Boolean (True/False) or 1/0
- `email_domain`: User's email domain
- `device`: User's device type
- `location`: User's location
- `timestamp`: When the step was attempted
""")

# File uploader
uploaded_file = st.sidebar.file_uploader(
    "Choose a CSV file",
    type=['csv'],
    help="Upload your onboarding funnel data"
)

# Main content
st.title("🚀 Onboarding Drop-off Analyzer")
st.markdown("Analyze user onboarding funnels and get AI-powered insights for improving conversion rates.")

if uploaded_file is not None:
    try:
        # Load and analyze data
        df = load_data(uploaded_file)
        
        if df is not None and not df.empty:
            st.success(f"✅ Data loaded successfully! {len(df)} records found.")
            
            # Display data preview
            with st.expander("📋 Data Preview", expanded=False):
                st.dataframe(df.head(10))
                st.write(f"**Data shape:** {df.shape}")
                st.write(f"**Columns:** {list(df.columns)}")
            
            # Analyze funnel
            funnel_data = analyze_funnel(df)
            
            if funnel_data:
                # Create funnel chart
                st.header("📈 Onboarding Funnel Analysis")
                
                # Funnel chart using plotly
                fig = go.Figure(go.Funnel(
                    y=funnel_data['step'],
                    x=funnel_data['users'],
                    textinfo="value+percent initial",
                    textposition="inside",
                    marker={"color": ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"],
                           "line": {"width": [4, 2, 2, 3, 1, 1], "color": ["wheat", "wheat", "blue", "wheat", "wheat", "wheat"]}},
                    connector={"line": {"color": "royalblue", "width": 3}}
                ))
                
                fig.update_layout(
                    title="User Drop-off at Each Onboarding Step",
                    width=800,
                    height=500,
                    showlegend=False
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Drop-off percentages table
                st.subheader("📊 Drop-off Statistics")
                dropoff_df = pd.DataFrame(funnel_data)
                dropoff_df['dropoff_rate'] = dropoff_df['dropoff_rate'].round(2)
                dropoff_df['retention_rate'] = dropoff_df['retention_rate'].round(2)
                st.dataframe(dropoff_df, use_container_width=True)
                
                # User clustering analysis
                st.header("🔍 User Drop-off Clustering")
                
                with st.spinner("Analyzing user clusters..."):
                    clusters = cluster_dropoffs(df)
                    
                if clusters:
                    st.success("✅ User clustering completed!")
                    
                    # Display cluster information
                    for step, cluster_info in clusters.items():
                        with st.expander(f"Step: {step}", expanded=False):
                            if cluster_info and 'clusters' in cluster_info:
                                st.write(f"**Number of clusters:** {len(cluster_info['clusters'])}")
                                st.write(f"**Total users who dropped off:** {cluster_info['total_dropoffs']}")
                                
                                # Show cluster details
                                for i, cluster in enumerate(cluster_info['clusters']):
                                    st.write(f"**Cluster {i+1}:** {cluster['size']} users")
                                    st.write(f"**Characteristics:** {cluster['characteristics']}")
                                    st.write("---")
                            else:
                                st.warning("No clustering data available for this step.")
                
                # LLM Analysis
                st.header("🤖 AI-Powered Insights")
                
                if st.button("Generate AI Insights", type="primary"):
                    with st.spinner("🤖 AI is analyzing your data..."):
                        # Prepare prompt for LLM
                        prompt = f"""
                        Analyze this onboarding funnel data and provide insights:
                        
                        Funnel Steps: {funnel_data['step']}
                        User Counts: {funnel_data['users']}
                        Drop-off Rates: {funnel_data['dropoff_rate']}
                        
                        Please provide:
                        1. **Key Hypotheses** about why users are dropping off at each step
                        2. **UX Recommendations** to improve conversion rates
                        3. **A/B Testing Ideas** to validate improvements
                        4. **Priority Actions** to focus on first
                        
                        Be specific and actionable in your recommendations.
                        """
                        
                        try:
                            llm_response = query_llama(prompt)
                            if llm_response:
                                st.success("✅ AI insights generated!")
                                st.markdown("### 🤖 AI Analysis Results")
                                st.markdown(llm_response)
                            else:
                                st.error("❌ Failed to generate AI insights. Please check your API key.")
                        except Exception as e:
                            st.error(f"❌ Error generating AI insights: {str(e)}")
                            st.info("💡 Make sure you have set your HF_API_KEY in the .env file")
                
                # Download results
                st.header("💾 Download Results")
                
                # Create summary report
                summary_data = {
                    'Step': funnel_data['step'],
                    'Users': funnel_data['users'],
                    'Drop-off Rate (%)': [f"{rate:.2f}%" for rate in funnel_data['dropoff_rate']],
                    'Retention Rate (%)': [f"{rate:.2f}%" for rate in funnel_data['retention_rate']]
                }
                
                summary_df = pd.DataFrame(summary_data)
                
                # Download buttons
                col1, col2 = st.columns(2)
                with col1:
                    st.download_button(
                        label="📥 Download Funnel Summary (CSV)",
                        data=summary_df.to_csv(index=False),
                        file_name="onboarding_funnel_summary.csv",
                        mime="text/csv"
                    )
                
                with col2:
                    st.download_button(
                        label="📥 Download Raw Data (CSV)",
                        data=df.to_csv(index=False),
                        file_name="onboarding_data_processed.csv",
                        mime="text/csv"
                    )
        
        else:
            st.error("❌ No data found in the uploaded file. Please check the file format.")
            
    except Exception as e:
        st.error(f"❌ Error processing file: {str(e)}")
        st.info("💡 Please ensure your CSV has the required columns: user_id, step, completed, email_domain, device, location, timestamp")

else:
    # Show sample data structure
    st.info("📁 Please upload a CSV file to begin analysis.")
    
    st.subheader("📋 Expected CSV Format")
    sample_data = {
        'user_id': ['user_001', 'user_001', 'user_002', 'user_002'],
        'step': ['signup', 'profile', 'signup', 'profile'],
        'completed': [True, False, True, True],
        'email_domain': ['gmail.com', 'gmail.com', 'yahoo.com', 'yahoo.com'],
        'device': ['mobile', 'mobile', 'desktop', 'desktop'],
        'location': ['US', 'US', 'UK', 'UK'],
        'timestamp': ['2024-01-01 10:00:00', '2024-01-01 10:05:00', '2024-01-01 11:00:00', '2024-01-01 11:05:00']
    }
    
    sample_df = pd.DataFrame(sample_data)
    st.dataframe(sample_df, use_container_width=True)
    
    st.markdown("""
    ### 🔧 Setup Instructions:
    1. **Install dependencies:** `pip install streamlit pandas plotly scikit-learn python-dotenv huggingface-hub`
    2. **Set API key:** Create a `.env` file with `HF_API_KEY=your_huggingface_token`
    3. **Run app:** `streamlit run app.py`
    """)

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit, Plotly, and AI")

