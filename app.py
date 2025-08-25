import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import os
from utils import query_llama
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
This tool analyzes any CSV file you upload.  
The AI agent will infer the structure and provide insights automatically.
""")

# File uploader
uploaded_file = st.sidebar.file_uploader(
    "Choose a CSV file",
    type=['csv'],
    help="Upload any CSV file for analysis"
)

# Main content
st.title("🚀 Onboarding Drop-off Analyzer")
st.markdown("Analyze user onboarding funnels and get AI-powered insights for improving conversion rates.")

if uploaded_file is not None:
    try:
        # Read CSV first to detect column types
        df = pd.read_csv(uploaded_file)
        
        # Convert any ID-like columns to string to avoid PyArrow conversion warnings
        id_columns = [col for col in df.columns if 'id' in col.lower() or col.lower() in ['id', 'user_id', 'userid']]
        for col in id_columns:
            df[col] = df[col].astype(str)
        
        if df is not None and not df.empty:
            st.success(f"✅ Data loaded successfully! {len(df)} records found.")
            
            # Organize initial data views with tabs
            tab1, tab2, tab3 = st.tabs(["📋 Data Preview", "📊 Data Summary", "🔎 Column Types"])
            with tab1:
                st.dataframe(df.head(10))
                st.write(f"**Data shape:** {df.shape}")
                st.write(f"**Columns:** {list(df.columns)}")
            with tab2:
                st.write(df.describe(include='all'))
                st.write("Unique values per column:")
                for col in df.columns:
                    st.write(f"- {col}: {df[col].nunique()} unique values")
            with tab3:
                for col in df.columns:
                    st.write(f"- {col}: {df[col].dtype}")

            # Data Cleaning Options
            st.header("🧹 Data Cleaning")
            drop_na = st.checkbox("Drop rows with missing values", value=False)
            if drop_na:
                df = df.dropna()
                st.info("Dropped rows with missing values.")

            # Let user select columns for funnel analysis
            st.header("🧩 Onboarding Funnel Analysis")
            st.markdown("Select columns for funnel analysis:")
            
            # Filter out user_id from step column options and set smart defaults
            step_options = [col for col in df.columns if col != 'user_id']
            user_options = df.columns
            
            # Set smart default indices
            step_default_idx = 0
            user_default_idx = 0
            
            # Try to find better defaults
            if 'step' in df.columns:
                step_default_idx = step_options.index('step')
            elif 'stage' in df.columns:
                step_default_idx = step_options.index('stage')
            elif 'phase' in df.columns:
                step_default_idx = step_options.index('phase')
                
            if 'user_id' in df.columns:
                user_default_idx = df.columns.get_loc('user_id')
            elif 'id' in df.columns:
                user_default_idx = df.columns.get_loc('id')
            
            step_col = st.selectbox("Step column", options=step_options, index=step_default_idx)
            user_col = st.selectbox("User ID column", options=user_options, index=user_default_idx)
            chart_type = st.selectbox("Chart type", ["Funnel", "Bar", "Pie"])

            if step_col and user_col:
                funnel_df = df.dropna(subset=[step_col, user_col])
                steps = funnel_df[step_col].dropna().unique()
                steps = sorted(steps, key=lambda x: list(funnel_df[step_col]).index(x))
                step_counts = funnel_df.groupby(step_col)[user_col].nunique().reindex(steps)
                total_users = step_counts.iloc[0] if not step_counts.empty else 0
                final_users = step_counts.iloc[-1] if not step_counts.empty else 0
                conversion_rate = (final_users / total_users * 100) if total_users else 0
                dropoff_rates = [f"{int((count/total_users)*100) if total_users else 0}%" for count in step_counts]

                # Funnel Visualization and Metrics (top)
                st.header("📊 Funnel Visualization")
                m1, m2, m3 = st.columns(3)
                m1.metric(label="Total Starting Users", value=f"{total_users}")
                m2.metric(label="Final Converted Users", value=f"{final_users}")
                m3.metric(label="Overall Conversion", value=f"{conversion_rate:.1f}%")

                # Define custom colors
                colors = ["#FF4B4B", "#4BFFB3", "#4B7BFF", "#FFD24B", "#FF4B9B", "#B34BFF", "#4BFFDA"]

                if chart_type == "Funnel":
                    fig = go.Funnel(
                        y=steps,
                        x=step_counts,
                        textinfo="value+percent initial",
                        marker={"color": colors[:len(steps)]}
                    )
                elif chart_type == "Bar":
                    fig = go.Bar(
                        x=steps,
                        y=step_counts,
                        marker_color=colors[:len(steps)]
                    )
                elif chart_type == "Pie":
                    fig = go.Pie(
                        labels=steps,
                        values=step_counts,
                        marker={"colors": colors[:len(steps)]}
                    )
                st.plotly_chart(go.Figure(fig), use_container_width=True)
                st.markdown("**User Drop-off at Each Onboarding Step**")
                for step, count, rate in zip(steps, step_counts, dropoff_rates):
                    st.markdown(f"- `{step}`: **{count} users** ({rate})")

                # Download Analysis Summary
                st.header("💾 Download Analysis Summary")
                summary_df = pd.DataFrame({
                    "Step": steps,
                    "User_Count": step_counts.values,
                    "Dropoff_Rate": dropoff_rates
                })
                st.download_button(
                    label="📥 Download Analysis Summary (CSV)",
                    data=summary_df.to_csv(index=False),
                    file_name="analysis_summary.csv",
                    mime="text/csv"
                )

                # AI Analysis (below graph)
                st.header("🤖 AI Analysis")
                custom_prompt = st.text_area(
                    "Custom AI prompt (optional):",
                    value="You are an expert UX and product analyst. Analyze the following CSV data and infer its structure, key columns, and possible insights.\n\nFormat your analysis as follows:\n\n## Analysis of Onboarding Funnel Data:\n\n1. **Key Hypotheses:**\n   - Bullet points for each hypothesis about user drop-off or friction.\n\n2. **UX Recommendations:**\n   - Bullet points with actionable UX improvements.\n\n3. **A/B Testing Ideas:**\n   - Bullet points for possible A/B tests to validate hypotheses.\n\n4. **Priority Actions:**\n   - Bullet points for the most important next steps.\n\nHere is a sample of the data:"
                )
                if st.button("Analyze with AI", key="ai_button"):
                    with st.spinner("AI agent is analyzing your data..."):
                        try:
                            csv_sample = df.head(20).to_csv(index=False)
                            prompt = f"{custom_prompt}\n{csv_sample}"
                            llm_response = query_llama(prompt)
                            if llm_response:
                                st.success("✅ AI insights generated!")
                                st.markdown(llm_response)
                                st.download_button(
                                    label="📥 Download AI Report (Markdown)",
                                    data=llm_response,
                                    file_name="ai_analysis.md",
                                    mime="text/markdown"
                                )
                            else:
                                st.error("❌ Failed to generate AI insights. Please check your API key.")
                        except Exception as e:
                            st.error(f"❌ Error generating AI insights: {str(e)}")
                            st.info("💡 Make sure you have set your HF_API_KEY in the .env file")

                # User Feedback
                st.header("💬 Feedback")
                feedback = st.text_area("Share your feedback or suggestions:")
                if st.button("Submit Feedback", key="feedback_button"):
                    st.success("Thank you for your feedback!")
            else:
                st.info("Select columns above to generate funnel analysis.")
        else:
            st.error("❌ No data found in the uploaded file. Please check the file format.")
    except Exception as e:
        st.error(f"❌ Error processing file: {str(e)}")
        st.info("💡 Please ensure your CSV is valid and not empty.")
else:
    st.info("📁 Please upload a CSV file to begin analysis.")
    st.markdown("""
    ### 🔧 Setup Instructions:
    1. **Install dependencies:** `pip install streamlit pandas plotly python-dotenv huggingface-hub`
    2. **Set API key:** Create a `.env` file with `HF_API_KEY=your_huggingface_token`
    3. **Run app:** `streamlit run app.py`
    """)

# Footer
st.markdown("---")
st.markdown("Built By Code Smashers")

# Add modern theme with custom CSS
st.markdown("""
    <style>
    body, .main, .block-container {
        background: linear-gradient(120deg, #232526 0%, #485563 100%);
        color: #F3F3F3;
    }
    h1, h2, h3, h4 {
        color: #FF4B4B;
        font-family: 'Segoe UI', 'Roboto', 'Arial', sans-serif;
        letter-spacing: 1px;
    }
    .stButton>button, .stDownloadButton>button {
        background: linear-gradient(90deg, #FF4B4B 0%, #FFB34B 100%);
        color: white;
        border-radius: 8px;
        font-weight: bold;
        box-shadow: 0 2px 8px rgba(0,0,0,0.15);
    }
    .stTextArea textarea {
        background-color: #222;
        color: #FFF;
        border-radius: 8px;
        font-size: 1rem;
    }
    .stSelectbox label {
        font-weight: bold;
        color: #FFB34B;
    }
    .stSidebar {
        background: linear-gradient(120deg, #232526 0%, #485563 100%);
        color: #FFF;
    }
    .stTabs [data-baseweb="tab-list"] {
        background: #232526;
        border-radius: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        color: #FFB34B;
        font-weight: bold;
        font-size: 1.1rem;
    }
    .stMetric {
        background: #232526;
        border-radius: 8px;
        color: #FFB34B;
        padding: 8px;
    }
    </style>
""", unsafe_allow_html=True)

