from backened import *


# Page configuration
st.set_page_config(
    page_title="Data Analysis Assistant",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #424242;
        margin-bottom: 1rem;
    }
    .insight-card {
        background-color: #f0f8ff;
        padding: 15px;
        border-radius: 5px;
        border-left: 5px solid #1E88E5;
        margin-bottom: 15px;
    }
    .file-uploader {
        padding: 2rem;
        border: 2px dashed #ddd;
        border-radius: 0.5rem;
        background-color: #fafafa;
    }
    .stTabs [data-baseweb="tab-panel"] {
        padding-top: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)

# Function to download dataframe as CSV
def get_download_link(df, filename="processed_data.csv"):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download CSV File</a>'
    return href

# Function to download HTML report
def get_html_download_link(html_content, filename="data_analysis_report.html"):
    b64 = base64.b64encode(html_content.encode()).decode()
    href = f'<a href="data:text/html;base64,{b64}" download="{filename}">Download HTML Report</a>'
    return href

# Initialize session state variables
if 'backend' not in st.session_state:
    st.session_state.backend = None
if 'df' not in st.session_state:
    st.session_state.df = None
if 'analysis_run' not in st.session_state:
    st.session_state.analysis_run = False
if 'report' not in st.session_state:
    st.session_state.report = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Header
st.markdown('<p class="main-header">Data Analysis Assistant</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Upload your data to get instant insights and visualizations</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("Upload Data")
    
    uploaded_file = st.file_uploader("Choose a CSV, Excel, or JSON file", 
                                    type=["csv", "xlsx", "xls", "json"],
                                    help="Upload your data file here. The app supports CSV, Excel, and JSON formats.")
    
    if uploaded_file is not None:
        try:
            # Determine file type and read accordingly
            file_type = uploaded_file.name.split('.')[-1].lower()
            
            if file_type == 'csv':
                df = pd.read_csv(uploaded_file)
            elif file_type in ['xlsx', 'xls']:
                df = pd.read_excel(uploaded_file)
            elif file_type == 'json':
                df = pd.read_json(uploaded_file)
            
            st.session_state.df = df
            st.session_state.backend = Backend(df)
            st.session_state.analysis_run = False
            st.success(f"File loaded successfully! {df.shape[0]} rows and {df.shape[1]} columns.")
            
            # Show data types
            st.write("Data Types:")
            data_types = pd.DataFrame({
                'Column': df.columns,
                'Type': df.dtypes.astype(str)}
            ).reset_index(drop=True)
            st.dataframe(data_types, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
    
    if st.session_state.df is not None:
        st.divider()
        st.header("Data Processing")
        
        # Run automatic data analysis
        if st.button("Run Automatic Analysis", type="primary"):
            with st.spinner("Analyzing data..."):
                # Generate data profile
                st.session_state.backend.generate_data_profile()
                
                # Generate insights
                st.session_state.backend.generate_insights()
                
                # Create default visualizations
                st.session_state.backend.generate_default_visualizations()
                
                st.session_state.analysis_run = True
                
                # Generate report
                st.session_state.report = st.session_state.backend.generate_report(
                    title=f"Analysis Report - {uploaded_file.name}"
                )
                
            st.success("Analysis completed!")
        
        # Download processed data
        if st.session_state.df is not None:
            st.markdown(get_download_link(st.session_state.df), unsafe_allow_html=True)
            
        # Download report if available
        if st.session_state.report is not None:
            html_report = st.session_state.backend.export_report_html(st.session_state.report)
            st.markdown(get_html_download_link(html_report), unsafe_allow_html=True)

# Main content
if st.session_state.df is not None:
    # Create tabs
    tabs = st.tabs(["Data View", "Exploratory Analysis", "Visualizations", "Chat Assistant", "Report"])
    
    # Data View Tab
    with tabs[0]:
        st.header("Data Preview")
        
        col1, col2 = st.columns(2)
        with col1:
            num_rows = st.slider("Number of rows to display", 5, 100, 10)
        with col2:
            show_stats = st.checkbox("Show quick statistics", value=True)
        
        st.dataframe(st.session_state.df.head(num_rows), use_container_width=True)
        
        if show_stats:
            st.subheader("Quick Statistics")
            
            numeric_cols = st.session_state.df.select_dtypes(include=['number']).columns.tolist()
            if numeric_cols:
                st.write("Numeric Columns:")
                st.dataframe(st.session_state.df[numeric_cols].describe(), use_container_width=True)
            
            cat_cols = st.session_state.df.select_dtypes(include=['object']).columns.tolist()
            if cat_cols and len(cat_cols) <= 10:  # Only show if there are not too many categorical columns
                st.write("Categorical Columns:")
                for col in cat_cols[:5]:  # Limit to first 5 columns
                    st.write(f"**{col}** - {st.session_state.df[col].nunique()} unique values")
                    st.dataframe(st.session_state.df[col].value_counts().head(5), use_container_width=True)
    
    # Exploratory Analysis Tab
    with tabs[1]:
        st.header("Exploratory Data Analysis")
        
        col1, col2 = st.columns(2)
        with col1:
            # Missing values analysis
            st.subheader("Missing Values")
            missing = st.session_state.df.isna().sum().reset_index()
            missing.columns = ['Column', 'Missing Count']
            missing['Missing %'] = (missing['Missing Count'] / len(st.session_state.df) * 100).round(2)
            missing = missing[missing['Missing Count'] > 0].sort_values('Missing Count', ascending=False)
            
            if not missing.empty:
                st.dataframe(missing, use_container_width=True)
                
                # Visualize missing values if there are any
                if len(missing) > 0:
                    fig = px.bar(missing, x='Column', y='Missing %', 
                                title='Missing Values by Column (%)',
                                height=400)
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No missing values found in your dataset.")
        
        with col2:
            # Duplicates analysis
            st.subheader("Duplicated Rows")
            duplicates = st.session_state.df.duplicated().sum()
            st.metric("Number of duplicated rows", duplicates)
            st.metric("Percentage of duplicated rows", f"{(duplicates / len(st.session_state.df) * 100):.2f}%")
            
            if duplicates > 0:
                if st.button("View duplicated rows"):
                    st.dataframe(st.session_state.df[st.session_state.df.duplicated(keep='first')], use_container_width=True)
        
        # Correlation Analysis
        st.subheader("Correlation Analysis")
        numeric_cols = st.session_state.df.select_dtypes(include=['number']).columns.tolist()
        
        if len(numeric_cols) >= 2:
            try:
                corr_matrix = st.session_state.df[numeric_cols].corr().round(2)
                
                # Show top correlations
                corr_pairs = []
                for i in range(len(numeric_cols)):
                    for j in range(i+1, len(numeric_cols)):
                        corr_value = corr_matrix.iloc[i, j]
                        if not np.isnan(corr_value):
                            corr_pairs.append({
                                'Column 1': numeric_cols[i],
                                'Column 2': numeric_cols[j],
                                'Correlation': corr_value
                            })
                
                corr_df = pd.DataFrame(corr_pairs).sort_values('Correlation', key=abs, ascending=False)
                
                col1, col2 = st.columns([2, 1])
                with col1:
                    # Interactive heatmap
                    fig = px.imshow(corr_matrix,
                                    labels=dict(color="Correlation"),
                                    x=corr_matrix.columns,
                                    y=corr_matrix.columns,
                                    color_continuous_scale='RdBu_r',
                                    zmin=-1, zmax=1)
                    fig.update_layout(height=500)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.write("Top Correlations:")
                    st.dataframe(corr_df.head(10), use_container_width=True)
                    
                    # Option to explore specific correlation
                    if len(corr_df) > 0:
                        selected_pair = st.selectbox(
                            "Explore correlation relationship:",
                            options=[f"{row['Column 1']} & {row['Column 2']} ({row['Correlation']:.2f})" for _, row in corr_df.head(10).iterrows()]
                        )
                        
                        if selected_pair:
                            col1_name = selected_pair.split(' & ')[0]
                            col2_name = selected_pair.split(' & ')[1].split(' (')[0]
                            
                            # Get correlation analysis for selected columns
                            correlation_result = st.session_state.backend._handle_correlation_query(
                                st.session_state.df, col1_name, col2_name
                            )
                            
                            st.write(correlation_result['text'])
                            if correlation_result['chart'] is not None:
                                st.plotly_chart(correlation_result['chart'], use_container_width=True)
            
            except Exception as e:
                st.error(f"Error calculating correlations: {str(e)}")
        else:
            st.info("Not enough numeric columns for correlation analysis.")

    # Visualizations Tab
    with tabs[2]:
        st.header("Data Visualizations")
        
        viz_type = st.selectbox(
            "Select visualization type:",
            ["Distribution Analysis", "Relationship Analysis", "Time Series Analysis", "Custom Query"]
        )
        
        if viz_type == "Distribution Analysis":
            cols = st.columns(2)
            with cols[0]:
                column = st.selectbox("Select column to analyze:", st.session_state.df.columns)
                
            with cols[1]:
                chart_type = st.selectbox(
                    "Select chart type:",
                    ["Histogram", "Box Plot", "Violin Plot", "Count Plot"],
                    key="dist_chart_type"
                )
            
            # Create visualization based on data type and chart selection
            try:
                if pd.api.types.is_numeric_dtype(st.session_state.df[column]):
                    if chart_type == "Histogram":
                        fig = px.histogram(st.session_state.df, x=column, marginal="box", 
                                          title=f"Distribution of {column}")
                    elif chart_type == "Box Plot":
                        fig = px.box(st.session_state.df, y=column, 
                                    title=f"Box Plot of {column}")
                    elif chart_type == "Violin Plot":
                        fig = px.violin(st.session_state.df, y=column, box=True, 
                                       title=f"Violin Plot of {column}")
                    else:  # Count Plot for numeric binned
                        fig = px.histogram(st.session_state.df, x=column, 
                                          title=f"Count Plot of {column}")
                else:  # Categorical
                    if chart_type == "Count Plot":
                        counts = st.session_state.df[column].value_counts().reset_index()
                        counts.columns = [column, 'count']
                        fig = px.bar(counts, x=column, y='count', 
                                    title=f"Count Plot of {column}")
                    else:
                        st.warning(f"Chart type {chart_type} not suitable for categorical data. Using Count Plot instead.")
                        counts = st.session_state.df[column].value_counts().reset_index()
                        counts.columns = [column, 'count']
                        fig = px.bar(counts, x=column, y='count', 
                                    title=f"Count Plot of {column}")
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Show stats for the column
                st.subheader(f"Statistics for {column}")
                
                if pd.api.types.is_numeric_dtype(st.session_state.df[column]):
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Mean", f"{st.session_state.df[column].mean():.2f}")
                    col2.metric("Median", f"{st.session_state.df[column].median():.2f}")
                    col3.metric("Std Dev", f"{st.session_state.df[column].std():.2f}")
                    col4.metric("Missing", f"{st.session_state.df[column].isna().sum()} ({st.session_state.df[column].isna().sum() / len(st.session_state.df) * 100:.1f}%)")
                else:
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Unique Values", st.session_state.df[column].nunique())
                    col2.metric("Most Common", f"{st.session_state.df[column].value_counts().index[0]}")
                    col3.metric("Missing", f"{st.session_state.df[column].isna().sum()} ({st.session_state.df[column].isna().sum() / len(st.session_state.df) * 100:.1f}%)")
                    
                    # Show top values
                    st.write("Top 5 values:")
                    st.dataframe(st.session_state.df[column].value_counts().head(5).reset_index())
                
            except Exception as e:
                st.error(f"Error creating visualization: {str(e)}")

        elif viz_type == "Relationship Analysis":
            cols = st.columns(3)
            with cols[0]:
                x_col = st.selectbox("Select X-axis column:", st.session_state.df.columns, key="x_col")
            with cols[1]:
                y_col = st.selectbox("Select Y-axis column:", st.session_state.df.columns, key="y_col")
            with cols[2]:
                chart_type = st.selectbox(
                    "Select chart type:",
                    ["Scatter Plot", "Line Plot", "Bar Chart", "Box Plot"],
                    key="rel_chart_type"
                )
            
            color_col = st.selectbox("Color by (optional):", 
                                   ["None"] + st.session_state.df.columns.tolist(),
                                   key="color_col")
            
            if color_col == "None":
                color_col = None
            
            # Create visualization based on selection
            try:
                if chart_type == "Scatter Plot":
                    fig = px.scatter(st.session_state.df, x=x_col, y=y_col, color=color_col,
                                    title=f"{y_col} vs {x_col}",
                                    trendline="ols" if pd.api.types.is_numeric_dtype(st.session_state.df[x_col]) and 
                                                      pd.api.types.is_numeric_dtype(st.session_state.df[y_col]) else None)
                
                elif chart_type == "Line Plot":
                    fig = px.line(st.session_state.df, x=x_col, y=y_col, color=color_col,
                                 title=f"{y_col} vs {x_col}")
                
                elif chart_type == "Bar Chart":
                    if pd.api.types.is_numeric_dtype(st.session_state.df[y_col]) and not pd.api.types.is_numeric_dtype(st.session_state.df[x_col]):
                        # Group by categorical X, calculate mean of numeric Y
                        agg_df = st.session_state.df.groupby(x_col)[y_col].mean().reset_index()
                        fig = px.bar(agg_df, x=x_col, y=y_col, color=color_col,
                                    title=f"Average {y_col} by {x_col}")
                    else:
                        fig = px.bar(st.session_state.df, x=x_col, y=y_col, color=color_col,
                                    title=f"{y_col} vs {x_col}")
                
                elif chart_type == "Box Plot":
                    if not pd.api.types.is_numeric_dtype(st.session_state.df[x_col]) and pd.api.types.is_numeric_dtype(st.session_state.df[y_col]):
                        fig = px.box(st.session_state.df, x=x_col, y=y_col, color=color_col,
                                    title=f"Distribution of {y_col} by {x_col}")
                    else:
                        st.warning("Box plots work best with categorical X and numeric Y. Attempting to create plot anyway.")
                        fig = px.box(st.session_state.df, x=x_col, y=y_col, color=color_col,
                                    title=f"Distribution of {y_col} by {x_col}")
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Calculate correlation for numeric columns
                if pd.api.types.is_numeric_dtype(st.session_state.df[x_col]) and pd.api.types.is_numeric_dtype(st.session_state.df[y_col]):
                    correlation = st.session_state.df[[x_col, y_col]].corr().iloc[0, 1]
                    st.metric("Correlation Coefficient", f"{correlation:.4f}")
                    
                    if abs(correlation) > 0.7:
                        st.info("Strong correlation detected!")
                    elif abs(correlation) > 0.3:
                        st.info("Moderate correlation detected.")
                    else:
                        st.info("Weak or no correlation.")
                
            except Exception as e:
                st.error(f"Error creating visualization: {str(e)}")

        elif viz_type == "Time Series Analysis":
            # Check for datetime columns
            datetime_cols = []
            for col in st.session_state.df.columns:
                try:
                    if pd.api.types.is_datetime64_any_dtype(st.session_state.df[col]) or \
                       pd.to_datetime(st.session_state.df[col], errors='coerce').notna().all():
                        datetime_cols.append(col)
                except:
                    pass
            
            if not datetime_cols:
                st.warning("No datetime columns detected in your dataset. Please convert a column to datetime first.")
            else:
                cols = st.columns(2)
                with cols[0]:
                    date_col = st.selectbox("Select date/time column:", datetime_cols)
                with cols[1]:
                    value_col = st.selectbox(
                        "Select value to analyze:", 
                        [col for col in st.session_state.df.columns if col != date_col and 
                         pd.api.types.is_numeric_dtype(st.session_state.df[col])]
                    )
                
                # Ensure date column is datetime
                if not pd.api.types.is_datetime64_any_dtype(st.session_state.df[date_col]):
                    with st.spinner("Converting to datetime..."):
                        st.session_state.df[date_col] = pd.to_datetime(st.session_state.df[date_col], errors='coerce')
                
                # Run time series analysis
                with st.spinner("Analyzing time series data..."):
                    time_series_result = st.session_state.backend._handle_time_series_query(
                        st.session_state.df, date_col, value_col
                    )
                
                st.write(time_series_result['text'])
                if time_series_result['chart'] is not None:
                    st.plotly_chart(time_series_result['chart'], use_container_width=True)
                    
                # Additional time series options
                with st.expander("Additional Time Series Options"):
                    agg_options = {
                        'Daily': 'D',
                        'Weekly': 'W',
                        'Monthly': 'M',
                        'Quarterly': 'Q',
                        'Yearly': 'Y'
                    }
                    
                    agg_period = st.selectbox("Resample time series by:", list(agg_options.keys()))
                    agg_func = st.selectbox("Aggregation function:", ["Mean", "Sum", "Min", "Max", "Count"])
                    
                    if st.button("Apply Resampling"):
                        try:
                            # Set index to date column for resampling
                            temp_df = st.session_state.df.copy()
                            temp_df[date_col] = pd.to_datetime(temp_df[date_col])
                            temp_df = temp_df.set_index(date_col)
                            
                            # Resample
                            agg_map = {"Mean": "mean", "Sum": "sum", "Min": "min", "Max": "max", "Count": "count"}
                            resampled = temp_df[value_col].resample(agg_options[agg_period]).agg(agg_map[agg_func])
                            
                            # Create figure
                            fig = px.line(
                                x=resampled.index, 
                                y=resampled.values,
                                title=f"{agg_func} of {value_col} ({agg_period})",
                                labels={"x": date_col, "y": f"{agg_func} of {value_col}"}
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Calculate period over period changes
                            if len(resampled) > 1:
                                resampled_pct = resampled.pct_change() * 100
                                
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.metric(
                                        "Latest Value", 
                                        f"{resampled.iloc[-1]:.2f}",
                                        f"{resampled.iloc[-1] - resampled.iloc[-2]:.2f}" if len(resampled) > 1 else None
                                    )
                                
                                with col2:
                                    st.metric(
                                        "Period-over-Period Change", 
                                        f"{resampled_pct.iloc[-1]:.2f}%" if len(resampled) > 1 else "N/A"
                                    )
                        
                        except Exception as e:
                            st.error(f"Error in time series resampling: {str(e)}")

        elif viz_type == "Custom Query":
            query = st.text_area(
                "Enter your query about the data:",
                placeholder="Example: Show me the correlation between sales and marketing spend",
                height=100
            )
            
            if st.button("Analyze", key="analyze_query"):
                with st.spinner("Processing query..."):
                    # Simple keyword-based query processor
                    query = query.lower()
                    
                    if "correlation" in query or "relationship" in query:
                        # Extract column names from query
                        columns = [col.lower() for col in st.session_state.df.columns]
                        
                        # Find mentioned columns
                        mentioned_cols = []
                        for col in columns:
                            if col.lower() in query:
                                mentioned_cols.append(col)
                        
                        if len(mentioned_cols) >= 2:
                            correlation_result = st.session_state.backend._handle_correlation_query(
                                st.session_state.df, mentioned_cols[0], mentioned_cols[1]
                            )
                            
                            st.write(correlation_result['text'])
                            if correlation_result['chart'] is not None:
                                st.plotly_chart(correlation_result['chart'], use_container_width=True)
                        else:
                            st.warning("Please specify two column names for correlation analysis.")
                    
                    elif any(x in query for x in ["top", "highest", "max", "maximum", "lowest", "min", "minimum"]):
                        # Extract column name from query
                        mentioned_col = None
                        for col in st.session_state.df.columns:
                            if col.lower() in query:
                                mentioned_col = col
                                break
                        
                        if mentioned_col:
                            top_values_result = st.session_state.backend._handle_top_values_query(
                                st.session_state.df, mentioned_col, query
                            )
                            
                            st.write(top_values_result['text'])
                            if top_values_result['chart'] is not None:
                                st.plotly_chart(top_values_result['chart'], use_container_width=True)
                        else:
                            st.warning("Please specify a column name for top/bottom values analysis.")
                    
                    elif any(x in query for x in ["time", "trend", "over time", "change", "historical"]):
                        # Look for datetime columns
                        datetime_cols = []
                        for col in st.session_state.df.columns:
                            try:
                                if pd.api.types.is_datetime64_any_dtype(st.session_state.df[col]) or \
                                pd.to_datetime(st.session_state.df[col], errors='coerce').notna().all():
                                    datetime_cols.append(col)
                            except:
                                pass
                        
                        value_cols = [col for col in st.session_state.df.columns if 
                                     pd.api.types.is_numeric_dtype(st.session_state.df[col])]
                        
                        if datetime_cols and value_cols:
                            # Try to find mentioned columns
                            date_col = datetime_cols[0]  # Default to first datetime column
                            for col in datetime_cols:
                                if col.lower() in query:
                                    date_col = col
                                    break
                            
                            value_col = value_cols[0]  # Default to first numeric column
                            for col in value_cols:
                                if col.lower() in query:
                                    value_col = col
                                    break
                            
                            time_series_result = st.session_state.backend._handle_time_series_query(
                                st.session_state.df, date_col, value_col
                            )
                            
                            st.write(time_series_result['text'])
                            if time_series_result['chart'] is not None:
                                st.plotly_chart(time_series_result['chart'], use_container_width=True)
                        else:
                            st.warning("Could not find appropriate date and value columns for time series analysis.")
                    
                    else:
                        st.info("I couldn't understand the specific analysis you're looking for. Try asking about correlation, top values, or trends over time.")

    # Chat Assistant Tab
    with tabs[3]:
        st.header("Chat with Your Data")
        
        # Add Gemini API key input in the sidebar
        # with st.sidebar:
        #     if st.session_state.df is not None:
        #         st.divider()
        #         st.header("AI Settings")
        #         gemini_api_key = st.text_input("Enter Gemini API Key (optional)", type="password", key="gemini_key")
        #         if gemini_api_key:
        #             st.session_state.backend.gemini_api_key = gemini_api_key
        #             try:
        #                 genai.configure(api_key=gemini_api_key)
        #                 st.session_state.backend.gemini_model = genai.GenerativeModel('gemini-pro')
        #                 st.session_state.backend.gemini_available = True
        #                 st.success("Gemini API connected successfully!")
        #             except Exception as e:
        #                 st.error(f"Failed to connect to Gemini: {str(e)}")
        #                 st.session_state.backend.gemini_available = False
        
        # Display chat history
        for i, (query, response) in enumerate(st.session_state.chat_history):
            col1, col2 = st.columns([1, 9])
            # with col1:
            #     st.image("https://cdn.pixabay.com/photo/2016/04/01/10/04/amusing-1299756_960_720.png", width=50)
            with col2:
                st.write(f"**You:** {query}")
            
            col1, col2 = st.columns([1, 9])
            with col1:
                st.image("https://cdn.pixabay.com/photo/2016/04/01/10/04/amusing-1299756_960_720.png", width=20)
            with col2:
                st.write(f"**Assistant:** {response['text']}")
                if response.get('chart') is not None:
                    st.plotly_chart(response['chart'], use_container_width=True)
        
        # Chat input
        user_query = st.text_input("Ask a question about your data:", key="chat_input")
        
        if user_query and user_query.strip():
            with st.spinner("Analyzing your question..."):
                # First try to handle with built-in analysis
                response = None
                
                # Simple keyword-based query processor
                query = user_query.lower()
                
                if "correlation" in query or "relationship" in query or "related" in query:
                    # Extract column names from query
                    columns = [col.lower() for col in st.session_state.df.columns]
                    
                    # Find mentioned columns
                    mentioned_cols = []
                    for col in columns:
                        if col.lower() in query:
                            mentioned_cols.append(col)
                    
                    if len(mentioned_cols) >= 2:
                        response = st.session_state.backend._handle_correlation_query(
                            st.session_state.df, mentioned_cols[0], mentioned_cols[1]
                        )
                
                elif any(x in query for x in ["top", "highest", "max", "maximum", "lowest", "min", "minimum"]):
                    # Extract column name from query
                    mentioned_col = None
                    for col in st.session_state.df.columns:
                        if col.lower() in query:
                            mentioned_col = col
                            break
                    
                    if mentioned_col:
                        response = st.session_state.backend._handle_top_values_query(
                            st.session_state.df, mentioned_col, query
                        )
                
                elif any(x in query for x in ["time", "trend", "over time", "change", "historical"]):
                    # Look for datetime columns
                    datetime_cols = []
                    for col in st.session_state.df.columns:
                        try:
                            if pd.api.types.is_datetime64_any_dtype(st.session_state.df[col]) or \
                            pd.to_datetime(st.session_state.df[col], errors='coerce').notna().all():
                                datetime_cols.append(col)
                        except:
                            pass
                    
                    value_cols = [col for col in st.session_state.df.columns if 
                                pd.api.types.is_numeric_dtype(st.session_state.df[col])]
                    
                    if datetime_cols and value_cols:
                        # Try to find mentioned columns
                        date_col = datetime_cols[0]  # Default to first datetime column
                        for col in datetime_cols:
                            if col.lower() in query:
                                date_col = col
                                break
                        
                        value_col = value_cols[0]  # Default to first numeric column
                        for col in value_cols:
                            if col.lower() in query:
                                value_col = col
                                break
                        
                        response = st.session_state.backend._handle_time_series_query(
                            st.session_state.df, date_col, value_col
                        )
                
                # If built-in analysis didn't handle it or we want to augment with Gemini
                if response is None and st.session_state.backend.gemini_available:
                    # Fall back to Gemini for more complex queries
                    response = st.session_state.backend.generate_ai_response(user_query)
                
                # If we still don't have a response (no Gemini or it failed)
                if response is None:
                    response = {
                        'text': "I can help you analyze your data. Try asking about:\n\n"
                                "- Correlations between columns\n"
                                "- Top/highest/lowest values in a column\n"
                                "- Trends or changes over time\n"
                                "- Summary statistics of your data\n"
                                "- Missing values analysis\n\n"
                                "Please include specific column names in your question for the best results.",
                        'chart': None
                    }
                
                # Add the query and response to chat history
                st.session_state.chat_history.append((user_query, response))
                
                # Force a rerun to show the updated chat history
                st.rerun()
                st.stop()

        # Report Tab
    with tabs[4]:
        st.header("Analysis Report")
        
        if not st.session_state.analysis_run:
            st.info("Run the automatic analysis from the sidebar to generate a report.")
        else:
            # Display the report
            if st.session_state.report:
                for section in st.session_state.report['sections']:
                    st.subheader(section['title'])
                    st.write(section['content'])
                    
                    for item in section['items']:
                        item_type = item.get('type', 'text')
                        
                        if item_type == 'text':
                            st.write(item['content'])
                            
                        elif item_type == 'table':
                            st.write(item.get('caption', ''))
                            st.write(item['data'], unsafe_allow_html=True)
                            
                        elif item_type == 'image':
                            st.write(item.get('caption', ''))
                            st.image(item['data'])
                            st.write(item.get('description', ''))
                            
                        elif item_type == 'insight':
                            with st.container():
                                st.markdown(f"""
                                <div class="insight-card">
                                    <h3>{item['title']}</h3>
                                    <p>{item['content']}</p>
                                </div>
                                """, unsafe_allow_html=True)
                
                # Download report option
                html_report = st.session_state.backend.export_report_html(st.session_state.report)
                st.download_button(
                    label="Download Full Report",
                    data=html_report,
                    file_name="data_analysis_report.html",
                    mime="text/html"
                )
else:
    # No data uploaded
    st.write("👈 Please upload a CSV, Excel, or JSON file in the sidebar to get started.")
    
    # Show example data sources
    with st.expander("Looking for sample datasets?"):
        st.write("""
        Here are some free data sources you can use:
        - [Kaggle Datasets](https://www.kaggle.com/datasets)
        - [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets.php)
        - [Google Dataset Search](https://datasetsearch.research.google.com/)
        - [Data.gov](https://www.data.gov/)
        """)
    
    # Show feature preview
    st.subheader("What this app can do")
    
    features = [
        {
            "title": "Automatic Data Analysis",
            "description": "Upload your data and get instant insights without writing code."
        },
        {
            "title": "Interactive Visualizations",
            "description": "Explore your data with beautiful interactive charts and graphs."
        },
        {
            "title": "Chat With Your Data",
            "description": "Ask questions about your data in natural language and get answers instantly."
        },
        {
            "title": "Comprehensive Reports",
            "description": "Generate professional data analysis reports that you can download and share."
        }
    ]
    
    col1, col2 = st.columns(2)
    
    for i, feature in enumerate(features):
        with col1 if i % 2 == 0 else col2:
            st.markdown(f"""
            <div style="padding: 1rem; border-radius: 0.5rem; border: 1px solid #ddd; margin-bottom: 1rem;">
                <h3>{feature['title']}</h3>
                <p>{feature['description']}</p>
            </div>
            """, unsafe_allow_html=True)

# Footer
st.markdown("""
<div style="text-align: center; margin-top: 3rem; padding: 1rem; border-top: 1px solid #ddd;">
    <p>Data Analysis Assistant | Built with Streamlit and Python</p>
</div>
""", unsafe_allow_html=True)