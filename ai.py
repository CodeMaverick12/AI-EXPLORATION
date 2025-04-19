import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import plotly.express as px
import plotly.graph_objects as go
import base64
from io import BytesIO
import tempfile
from transformers import pipeline
import torch
import datetime
import re

from PIL import Image
import io
import speech_recognition as sr

# Set page config
st.set_page_config(
    page_title="DataQuest AI Explorer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'cleaned_df' not in st.session_state:
    st.session_state.cleaned_df = None
if 'insights' not in st.session_state:
    st.session_state.insights = []
if 'visualizations' not in st.session_state:
    st.session_state.visualizations = []
if 'report_components' not in st.session_state:
    st.session_state.report_components = []
if 'agent_active' not in st.session_state:
    st.session_state.agent_active = False
if 'agent_findings' not in st.session_state:
    st.session_state.agent_findings = {}

# Load Local LLM (T5 small)
@st.cache_resource
def load_local_llm():
    return pipeline("text2text-generation", model="google/flan-t5-small", device=-1 if torch.cuda.is_available() else -1)

# Function to clean dataset
def clean_dataset(df):
    # Make a copy to avoid modifying original data
    cleaned = df.copy()
    
    # Drop rows with too many missing values (>50%)
    cleaned = cleaned.dropna(thresh=len(cleaned.columns) * 0.5)
    
    # For numeric columns: fill missing with median
    numeric_cols = cleaned.select_dtypes(include=['float64', 'int64']).columns
    for col in numeric_cols:
        cleaned[col] = cleaned[col].fillna(cleaned[col].median())
    
    # For categorical columns: fill missing with mode
    cat_cols = cleaned.select_dtypes(include=['object']).columns
    for col in cat_cols:
        if cleaned[col].isna().sum() > 0:
            mode_val = cleaned[col].mode()[0]
            cleaned[col] = cleaned[col].fillna(mode_val)
    
    # Convert date-like columns to datetime
    for col in cleaned.columns:
        if cleaned[col].dtype == 'object':
            # Try to convert to datetime if it looks like a date
            try:
                cleaned[col] = pd.to_datetime(cleaned[col])
            except:
                pass
    
    return cleaned

# Function to profile dataset
def profile_dataset(df):
    profile = {}
    
    # Basic statistics
    profile['row_count'] = len(df)
    profile['column_count'] = len(df.columns)
    profile['missing_values'] = df.isna().sum().sum()
    profile['duplicate_rows'] = df.duplicated().sum()
    
    # Column-level statistics
    profile['columns'] = {}
    for col in df.columns:
        col_stats = {}
        col_stats['dtype'] = str(df[col].dtype)
        col_stats['missing'] = df[col].isna().sum()
        
        if pd.api.types.is_numeric_dtype(df[col]):
            col_stats['min'] = float(df[col].min())
            col_stats['max'] = float(df[col].max())
            col_stats['mean'] = float(df[col].mean())
            col_stats['median'] = float(df[col].median())
            col_stats['std'] = float(df[col].std())
        elif pd.api.types.is_object_dtype(df[col]):
            col_stats['unique_values'] = int(df[col].nunique())
            col_stats['most_common'] = df[col].value_counts().index[0] if len(df[col].value_counts()) > 0 else None
        
        profile['columns'][col] = col_stats
    
    return profile

# Function to generate initial insights
def generate_initial_insights(df, profile):
    insights = {}
    
    # Narrative summary
    summary = f"The dataset contains {profile['row_count']} rows and {profile['column_count']} columns. "
    
    if profile['missing_values'] > 0:
        summary += f"There are {profile['missing_values']} missing values in the dataset. "
    
    if profile['duplicate_rows'] > 0:
        summary += f"Found {profile['duplicate_rows']} duplicate rows. "
    
    # Identify highly correlated numeric columns
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    if len(numeric_cols) >= 2:
        corr_matrix = df[numeric_cols].corr().abs()
        high_corr = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i):
                if corr_matrix.iloc[i, j] > 0.7:
                    high_corr.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]))
        
        if high_corr:
            summary += f"Found {len(high_corr)} pairs of highly correlated numeric features. "
    
    # Identify columns with high cardinality
    cat_cols = df.select_dtypes(include=['object']).columns
    high_card_cols = []
    for col in cat_cols:
        if df[col].nunique() > 50:
            high_card_cols.append(col)
    
    if high_card_cols:
        summary += f"Columns with high cardinality: {', '.join(high_card_cols)}. "
    
    # Identify potential outliers in numeric columns
    outliers_info = []
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col].count()
        if outliers > 0:
            outliers_info.append((col, outliers))
    
    if outliers_info:
        summary += "Outliers detected in some numeric columns. "
    
    insights['summary'] = summary
    insights['high_correlations'] = high_corr
    insights['outliers'] = outliers_info
    
    return insights

# Function to create visualizations based on data type
def create_visualizations(df, profile):
    visualizations = []
    
    # 1. For numeric columns: histogram with KDE
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    if len(numeric_cols) > 0:
        for col in numeric_cols[:3]:  # Limit to first 3 columns for better performance
            fig = px.histogram(df, x=col, marginal="box", title=f"Distribution of {col}")
            visualizations.append({
                'title': f"Distribution of {col}",
                'type': 'histogram',
                'figure': fig
            })
    
    # 2. For categorical columns: bar chart
    cat_cols = df.select_dtypes(include=['object']).columns
    if len(cat_cols) > 0:
        for col in cat_cols[:3]:  # Limit to first 3 columns
            # Only use columns with reasonable number of unique values
            if df[col].nunique() <= 20:
                value_counts = df[col].value_counts().reset_index()
                value_counts.columns = [col, 'count']
                fig = px.bar(value_counts, x=col, y='count', title=f"Counts of {col}")
                visualizations.append({
                    'title': f"Counts of {col}",
                    'type': 'bar',
                    'figure': fig
                })
    
    # 3. Correlation matrix for numeric columns
    if len(numeric_cols) >= 2:
        fig = px.imshow(df[numeric_cols].corr(), title="Correlation Matrix")
        visualizations.append({
            'title': "Correlation Matrix",
            'type': 'heatmap',
            'figure': fig
        })
    
    # 4. Time series if date column exists
    date_cols = df.select_dtypes(include=['datetime64']).columns
    if len(date_cols) > 0 and len(numeric_cols) > 0:
        date_col = date_cols[0]
        num_col = numeric_cols[0]
        # Group by date and aggregate
        try:
            df_ts = df.groupby(df[date_col].dt.date)[num_col].mean().reset_index()
            fig = px.line(df_ts, x=date_col, y=num_col, title=f"{num_col} over time")
            visualizations.append({
                'title': f"{num_col} over time",
                'type': 'line',
                'figure': fig
            })
        except:
            pass
    
    return visualizations

# Function to handle natural language queries
def process_nlq(query, df):
    # Use LLM to interpret the query
    llm = load_local_llm()
    columns_info = ", ".join([f"{col} ({df[col].dtype})" for col in df.columns])
    
    prompt = f"""
    Based on the dataset with columns: {columns_info}
    
    User query: "{query}"
    
    Identify what type of analysis the user wants:
    1. If they want to see a distribution or count, suggest a histogram or bar chart for the relevant column
    2. If they want to compare two variables, suggest a scatter plot or bar chart
    3. If they want to see trends over time, suggest a line chart
    4. If they want statistics, suggest calculating mean, median, etc.
    
    Be specific about which columns to use and what type of visualization to create.
    """
    
    response = llm(prompt, max_length=150)[0]['generated_text']
    
    # Extract key information from the response
    vis_type = None
    if "histogram" in response.lower():
        vis_type = "histogram"
    elif "bar chart" in response.lower():
        vis_type = "bar"
    elif "scatter" in response.lower():
        vis_type = "scatter"
    elif "line" in response.lower():
        vis_type = "line"
    
    # Try to extract column names from the response
    columns = []
    for col in df.columns:
        if col.lower() in response.lower():
            columns.append(col)
    
    # Generate the visualization or analysis
    result = {
        'query': query,
        'interpretation': response,
        'visualization': None,
        'analysis': None
    }
    
    if vis_type and len(columns) > 0:
        if vis_type == "histogram" and len(columns) > 0:
            fig = px.histogram(df, x=columns[0], title=f"Distribution of {columns[0]}")
            result['visualization'] = fig
        
        elif vis_type == "bar" and len(columns) > 0:
            if df[columns[0]].dtype == 'object' or df[columns[0]].nunique() < 20:
                value_counts = df[columns[0]].value_counts().reset_index()
                value_counts.columns = [columns[0], 'count']
                fig = px.bar(value_counts, x=columns[0], y='count', title=f"Counts of {columns[0]}")
                result['visualization'] = fig
            elif len(columns) > 1:
                fig = px.bar(df, x=columns[0], y=columns[1], title=f"{columns[1]} by {columns[0]}")
                result['visualization'] = fig
        
        elif vis_type == "scatter" and len(columns) >= 2:
            fig = px.scatter(df, x=columns[0], y=columns[1], title=f"{columns[1]} vs {columns[0]}")
            result['visualization'] = fig
        
        elif vis_type == "line" and len(columns) >= 1:
            if df[columns[0]].dtype == 'datetime64':
                if len(columns) >= 2:
                    fig = px.line(df, x=columns[0], y=columns[1], title=f"{columns[1]} over time")
                else:
                    # Try to find a numeric column
                    num_col = df.select_dtypes(include=['float64', 'int64']).columns[0]
                    fig = px.line(df, x=columns[0], y=num_col, title=f"{num_col} over time")
                result['visualization'] = fig
    
    # If we couldn't determine visualization, provide statistical analysis
    if not result['visualization']:
        analysis_text = "Unable to determine the appropriate visualization. "
        
        # Provide basic stats for the columns mentioned
        if columns:
            analysis_text += "Here are some basic statistics:\n\n"
            for col in columns:
                analysis_text += f"**{col}**:\n"
                if pd.api.types.is_numeric_dtype(df[col]):
                    analysis_text += f"- Mean: {df[col].mean():.2f}\n"
                    analysis_text += f"- Median: {df[col].median():.2f}\n"
                    analysis_text += f"- Min: {df[col].min():.2f}\n"
                    analysis_text += f"- Max: {df[col].max():.2f}\n"
                else:
                    analysis_text += f"- Most common value: {df[col].value_counts().index[0]}\n"
                    analysis_text += f"- Number of unique values: {df[col].nunique()}\n"
        else:
            analysis_text += "Couldn't identify relevant columns for your question."
        
        result['analysis'] = analysis_text
    
    return result

# Function to activate insight agent
def run_insight_agent(df):
    findings = {}
    
    # 1. Profile the dataset
    findings['profile'] = profile_dataset(df)
    
    # 2. Detect patterns
    patterns = []
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    
    # a. Detect outliers
    if len(numeric_cols) >= 1:
        model = IsolationForest(contamination=0.05, random_state=42)
        outlier_cols = numeric_cols.tolist()
        if len(outlier_cols) > 0:
            # Handle missing values for outlier detection
            X = df[outlier_cols].fillna(df[outlier_cols].mean())
            # Scale the data
            X_scaled = StandardScaler().fit_transform(X)
            # Predict outliers
            outlier_predictions = model.fit_predict(X_scaled)
            outlier_count = (outlier_predictions == -1).sum()
            if outlier_count > 0:
                patterns.append({
                    "type": "outliers",
                    "description": f"Found {outlier_count} potential outliers in the dataset.",
                    "details": "These might represent anomalies or unusual patterns.",
                    "visualization": None
                })
    
    # b. Detect clusters
    if len(numeric_cols) >= 2:
        cluster_cols = numeric_cols.tolist()
        if len(cluster_cols) > 0:
            # Handle missing values for clustering
            X = df[cluster_cols].fillna(df[cluster_cols].mean())
            # Scale the data
            X_scaled = StandardScaler().fit_transform(X)
            # Reduce dimensions for visualization if needed
            if len(cluster_cols) > 2:
                pca = PCA(n_components=2, random_state=42)
                X_pca = pca.fit_transform(X_scaled)
                explained_var = pca.explained_variance_ratio_.sum()
                # Perform K-means clustering
                kmeans = KMeans(n_clusters=min(5, len(df) // 10), random_state=42)
                clusters = kmeans.fit_predict(X_scaled)
                
                # Create visualization
                pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
                pca_df['Cluster'] = clusters
                fig = px.scatter(pca_df, x='PC1', y='PC2', color='Cluster', 
                                title=f"Cluster Analysis (PCA explains {explained_var:.2%} of variance)")
                
                patterns.append({
                    "type": "clusters",
                    "description": f"Identified {kmeans.n_clusters} distinct clusters in the data.",
                    "details": f"These clusters might represent different segments or groups within your data.",
                    "visualization": fig
                })
    
    # c. Detect correlations
    if len(numeric_cols) >= 2:
        corr_matrix = df[numeric_cols].corr().abs()
        high_corr = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i):
                if corr_matrix.iloc[i, j] > 0.7:
                    high_corr.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]))
        
        if high_corr:
            top_corr = sorted(high_corr, key=lambda x: x[2], reverse=True)[:5]
            
            # Create visualization for top correlation
            top_pair = top_corr[0]
            fig = px.scatter(df, x=top_pair[0], y=top_pair[1], 
                           title=f"Correlation: {top_pair[0]} vs {top_pair[1]} (r={top_pair[2]:.2f})")
            
            patterns.append({
                "type": "correlations",
                "description": f"Found {len(high_corr)} pairs of highly correlated features.",
                "details": "Top correlation: " + 
                           ", ".join([f"{x[0]} & {x[1]} (r={x[2]:.2f})" for x in top_corr]),
                "visualization": fig
            })
    
    # 3. Generate recommendations
    recommendations = []
    
    # a. Based on data quality
    missing_vals = df.isna().sum()
    high_missing = missing_vals[missing_vals > len(df) * 0.2]
    if len(high_missing) > 0:
        recommendations.append({
            "type": "data_quality",
            "title": "Address Missing Values",
            "description": f"Consider addressing columns with high missing values: {', '.join(high_missing.index)}",
            "action": "You could impute these values or drop these columns if they're not critical."
        })
    
    # b. Based on patterns
    if any(p["type"] == "outliers" for p in patterns):
        recommendations.append({
            "type": "analysis",
            "title": "Investigate Outliers",
            "description": "Your dataset contains outliers that might be skewing your analysis.",
            "action": "Consider investigating these outliers to determine if they represent errors or valid extreme values."
        })
    
    if any(p["type"] == "clusters" for p in patterns):
        recommendations.append({
            "type": "analysis",
            "title": "Segment Analysis",
            "description": "Your data shows distinct clusters that might represent different segments.",
            "action": "Consider analyzing each cluster separately to understand segment-specific characteristics."
        })
    
    # c. Based on correlations
    if any(p["type"] == "correlations" for p in patterns):
        recommendations.append({
            "type": "feature_engineering",
            "title": "Feature Selection",
            "description": "Your dataset contains highly correlated features which might be redundant.",
            "action": "Consider selecting one feature from each correlated pair to reduce dimensionality and improve model performance."
        })
    
    findings['patterns'] = patterns
    findings['recommendations'] = recommendations
    
    return findings

# Function to convert a figure to an image
def fig_to_image(fig):
    # Convert plotly figure to image
    img_bytes = fig.to_image(format="png")
    return Image.open(io.BytesIO(img_bytes))

# Function to create a PDF report
def generate_pdf_report(df, profile, insights, visualizations, agent_findings):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.html') as tmpfile:
        # Create HTML content
        html_content = f"""
        <html>
        <head>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    line-height: 1.6;
                    margin: 40px;
                }}
                h1, h2, h3 {{
                    color: #4a4a4a;
                }}
                table {{
                    border-collapse: collapse;
                    width: 100%;
                    margin-bottom: 20px;
                }}
                th, td {{
                    border: 1px solid #ddd;
                    padding: 8px;
                    text-align: left;
                }}
                th {{
                    background-color: #f2f2f2;
                }}
                .insight-box {{
                    background-color: #f9f9f9;
                    border-left: 4px solid #2196F3;
                    padding: 15px;
                    margin-bottom: 20px;
                }}
                .recommendation-box {{
                    background-color: #f9f9f9;
                    border-left: 4px solid #4CAF50;
                    padding: 15px;
                    margin-bottom: 20px;
                }}
            </style>
        </head>
        <body>
            <h1>Data Insights Report</h1>
            <p>Generated on {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
            
            <h2>Dataset Overview</h2>
            <p>Rows: {profile['row_count']}</p>
            <p>Columns: {profile['column_count']}</p>
            <p>Missing Values: {profile['missing_values']}</p>
            <p>Duplicate Rows: {profile['duplicate_rows']}</p>
            
            <h2>Data Summary</h2>
            <div class="insight-box">
                {insights['summary']}
            </div>
            
            <h2>Data Sample</h2>
            {df.head(5).to_html()}
            
            <h2>Column Summary</h2>
            <table>
                <tr>
                    <th>Column</th>
                    <th>Type</th>
                    <th>Missing</th>
                    <th>Details</th>
                </tr>
        """
        
        # Add column summary rows
        for col, stats in profile['columns'].items():
            html_content += f"""
                <tr>
                    <td>{col}</td>
                    <td>{stats['dtype']}</td>
                    <td>{stats['missing']}</td>
                    <td>
            """
            if 'mean' in stats:
                html_content += f"Mean: {stats['mean']:.2f}, "
                html_content += f"Median: {stats['median']:.2f}, "
                html_content += f"Min: {stats['min']:.2f}, "
                html_content += f"Max: {stats['max']:.2f}"
            elif 'unique_values' in stats:
                html_content += f"Unique values: {stats['unique_values']}"
                if stats['most_common']:
                    html_content += f", Most common: {stats['most_common']}"
            html_content += """
                    </td>
                </tr>
            """
        
        html_content += """
            </table>
        """
        
        # Add agent findings if available
        if agent_findings:
            html_content += """
                <h2>Key Patterns & Insights</h2>
            """
            
            if 'patterns' in agent_findings:
                for pattern in agent_findings['patterns']:
                    html_content += f"""
                        <div class="insight-box">
                            <h3>{pattern['type'].title()}</h3>
                            <p>{pattern['description']}</p>
                            <p>{pattern['details']}</p>
                        </div>
                    """
            
            if 'recommendations' in agent_findings:
                html_content += """
                    <h2>Recommendations</h2>
                """
                for rec in agent_findings['recommendations']:
                    html_content += f"""
                        <div class="recommendation-box">
                            <h3>{rec['title']}</h3>
                            <p>{rec['description']}</p>
                            <p><strong>Suggested Action:</strong> {rec['action']}</p>
                        </div>
                    """
        
        html_content += """
            </body>
            </html>
        """
        
        tmpfile.write(html_content.encode('utf-8'))
        
        # Convert HTML to PDF
        pdf_file = weasyprint.HTML(tmpfile.name).write_pdf()
        
        return pdf_file

# Function for voice recognition
def recognize_speech():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.write("Listening...")
        audio = r.listen(source)
        
    try:
        text = r.recognize_google(audio)
        st.success(f"Recognized: {text}")
        return text
    except sr.UnknownValueError:
        st.error("Could not understand audio")
        return None
    except sr.RequestError:
        st.error("Could not request results from speech recognition service")
        return None

# Main app
def main():
    st.title("📊 DataQuest AI Explorer")
    st.markdown("""
    ## AI-Powered Data Exploration Platform
    Upload your dataset and let our AI assistant help you discover insights without writing code!
    """)
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    tabs = ["Upload Data", "Data Profiler", "Smart Visualizations", 
            "Natural Language Querying", "Insight Agent", "Report Generator"]
    
    page = st.sidebar.radio("Go to", tabs)
    
    # File upload section
    if page == "Upload Data":
        st.header("📤 Upload Your Dataset")
        
        # Sample dataset option
        st.subheader("Choose a Sample Dataset or Upload Your Own")
        sample_option = st.selectbox(
            "Select a sample dataset or upload your own below:",
            ["Upload my own", "Hotel Reviews", "E-Commerce Complaints", "Education Feedback", "Public Opinion Survey"]
        )
        
        if sample_option != "Upload my own":
            if sample_option == "Hotel Reviews":
                df = pd.DataFrame({
                    'guest_id': range(1, 101),
                    'hotel_name': np.random.choice(['Grand Hotel', 'Seaside Resort', 'Mountain Lodge', 'City Center'], 100),
                    'rating': np.random.randint(1, 6, 100),
                    'location_rating': np.random.randint(1, 6, 100),
                    'cleanliness_rating': np.random.randint(1, 6, 100),
                    'service_rating': np.random.randint(1, 6, 100),
                    'value_rating': np.random.randint(1, 6, 100),
                    'review_date': pd.date_range(start='1/1/2023', periods=100),
                    'stay_duration': np.random.randint(1, 8, 100),
                    'trip_type': np.random.choice(['Business', 'Leisure', 'Family', 'Couple'], 100),
                    'review_text': np.random.choice([
                        'Great hotel with excellent service',
                        'Room was dirty and staff was unhelpful',
                        'Amazing location but overpriced',
                        'Perfect stay, will come back again',
                        'Mediocre experience overall'
                    ], 100)
                })
            elif sample_option == "E-Commerce Complaints":
                df = pd.DataFrame({
                    'complaint_id': range(1, 101),
                    'customer_id': np.random.randint(1000, 9999, 100),
                    'product_category': np.random.choice(['Electronics', 'Clothing', 'Home & Kitchen', 'Beauty', 'Books'], 100),
                    'complaint_type': np.random.choice(['Delivery Delay', 'Damaged Product', 'Wrong Item', 'Quality Issue', 'Customer Service'], 100),
                    'severity': np.random.randint(1, 6, 100),
                    'complaint_date': pd.date_range(start='1/1/2023', periods=100),
                    'resolution_days': np.random.randint(0, 30, 100),
                    'refund_amount': np.random.uniform(0, 200, 100),
                    'customer_region': np.random.choice(['North', 'South', 'East', 'West', 'Central'], 100),
                    'resolved': np.random.choice([True, False], 100)
                })
            elif sample_option == "Education Feedback":
                df = pd.DataFrame({
                    'student_id': range(1, 101),
                    'grade': np.random.choice(['Grade 7', 'Grade 8', 'Grade 9', 'Grade 10', 'Grade 11', 'Grade 12'], 100),
                    'subject': np.random.choice(['Math', 'Science', 'English', 'History', 'Art', 'Physical Education'], 100),
                    'teacher_rating': np.random.randint(1, 6, 100),
                    'content_rating': np.random.randint(1, 6, 100),
                    'difficulty_level': np.random.randint(1, 6, 100),
                    'attendance_percentage': np.random.uniform(70, 100, 100),
                    'survey_date': pd.date_range(start='9/1/2023', periods=100),
                    'gender': np.random.choice(['Male', 'Female', 'Other'], 100),
                    'extracurricular': np.random.choice([True, False], 100)
                })
            # else:  # Public Opinion Survey
            #     df = pd.DataFrame({
            #         'respondent_id': range(1, 101),
            #         'age_group': np.random.choice(['18-24', '25-34', '35-44', '45-54', '55-64', '65+'], 100),
            #         'gender': np.random.choice(['Male', 'Female', 'Other'], 100),
            #         'education_level': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD', 'Other'], 100),
            #         'region': np.