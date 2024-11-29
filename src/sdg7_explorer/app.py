import streamlit as st
import polars as pl
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import altair as alt
import pandas as pd


# Update the path to resources
RESOURCE_PATH = Path(__file__).parent.parent.parent / "resources"

@st.cache_data
def load_data():
    """Load and prepare the datasets."""
    # Load the investment data
    public_investment = pl.read_csv(
        RESOURCE_PATH / "csv" / "public_investment.csv",
        encoding='latin-1',
        ignore_errors=True,
        null_values=[".."],
        infer_schema_length=None
    )
    
    # First rename the columns
    public_investment = public_investment.with_columns([
        pl.col("Country/Area").alias("Economy"),
        pl.col("Amount (2020 USD million)").alias("DataValue")
    ]).drop(["Country/Area", "Amount (2020 USD million)"])
    
    # Then select the columns we want
    public_investment = public_investment.select([
        "ISO-code",
        "Economy",
        "Region",
        "Year",
        "Category",
        "Technology",
        "DataValue"
    ])
    
    # Load income groups data
    income_groups = pl.read_csv(
        RESOURCE_PATH / "data" / "CLASS.csv", 
        separator=";",
        encoding='latin-1'
    )
    
    # Fix the BOM marker in column name
    income_groups = income_groups.rename({
        "ï»¿Economy": "Economy"
    })
    
    return public_investment, income_groups

def clean_data(df):
    """Clean and prepare the data for analysis."""
    df_cleaned = df.filter(
        (pl.col("DataValue") > 0) &
        (pl.col("Category") == "Renewables")
    )
    
    return df_cleaned.group_by([
        "Economy",
        "Region",
        "Year"
    ]).agg([
        pl.col("DataValue").sum()
    ])

def process_investment_data(df):
    """Process the investment data for analysis."""
    # Group by relevant columns and sum the amount
    processed = (df
        .group_by(["Year", "Region", "Category"])
        .agg(
            pl.col("Amount (2020 USD million)").sum().alias("Total Investment")
        )
        .sort("Year")
    )
    
    return processed

def global_progress_visualization(df, title):
    """Create an enhanced area plot visualization of global progress."""
    # Prepare data
    world_data = (df
        .group_by(["Year", "Region"])
        .agg(
            pl.col("DataValue").sum().alias("Investment")
        )
        .sort("Year")
    ).to_pandas()
    
    # Custom color palette
    color_palette = {
        'East Asia & Pacific': '#1f77b4',      # Strong blue
        'Europe': '#2ca02c',                   # Forest green
        'North America': '#ff7f0e',            # Orange
        'Latin America & Caribbean': '#d62728', # Red
        'South Asia': '#9467bd',               # Purple
        'Sub-Saharan Africa': '#8c564b',       # Brown
        'Middle East & North Africa': '#e377c2',# Pink
        'Central Asia': '#7f7f7f',             # Gray
        'Oceania': '#bcbd22',                  # Olive
        'Other': '#17becf'                     # Cyan
    }
    
    # Create visualization with custom colors
    fig = px.area(
        world_data,
        x="Year",
        y="Investment",
        color="Region",
        title=f"Global {title} Trend by Region",
        template="plotly_dark",
        color_discrete_map=color_palette
    )
    
    # Enhance the figure layout
    fig.update_layout(
        height=500,
        yaxis_title="Investment (USD Million)",
        hovermode="x unified",
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(17, 17, 17, 0.8)",
            bordercolor="rgba(255, 255, 255, 0.3)",
            borderwidth=1,
            font=dict(color="white")
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Add key insights
    with st.expander("Key Insights"):
        st.markdown("""
        - Investment trends show significant regional variations
        - Overall upward trend in renewable energy investment
        - Some regions show more consistent growth than others
        """)

def income_based_disparities(df, income_groups, title):
    """Create visualization of disparities based on income groups."""
    df_with_income = df.join(
        income_groups,
        left_on="Economy",
        right_on="Economy",
        how="inner"
    )
    
    grouped_data = df_with_income.group_by(
        ["Year", "Income group"]
    ).agg(
        pl.col("DataValue").mean()
    ).sort("Year")
    
    fig = px.line(
        grouped_data.to_pandas(),
        x="Year",
        y="DataValue",
        color="Income group",
        title=f"{title} by Income Group"
    )
    
    fig.update_layout(yaxis_title=title)
    return fig

def regional_analysis_choropleth(df, income_groups, title):
    """Create choropleth map visualization with 5-year rolling investment totals"""
    
    # Merge data with income groups and ensure ISO codes are present
    df_with_region = df.join(
        income_groups, 
        left_on="Economy", 
        right_on="Economy", 
        how="inner"
    )
    
    # Add explanation about the 5-year window
    st.info("""
        📊 This visualization shows 5-year cumulative investments.
        For example, the 2020 value represents the total investment from 2016 to 2020.
        This helps to smooth out year-to-year variations and show sustained investment patterns.
    """)
    
    # Create tabs for different views
    tab1, tab2 = st.tabs(["Current Period", "Historical Snapshots"])
    
    with tab1:
        # Get the latest year
        latest_year = df_with_region["Year"].max()
        start_year = latest_year - 4  # 5 year window
        
        # Calculate 5-year sum for the latest period
        current_data = (df_with_region
            .filter(pl.col("Year").is_between(start_year, latest_year))
            .group_by(["ISO-code", "Economy", "Region"])
            .agg([
                pl.col("DataValue").sum().alias("DataValue")
            ])
            .with_columns([
                (pl.col("DataValue") / 1000).alias("DataValue_B")
            ])
        )
        
        create_choropleth(current_data, f"5-Year Cumulative Investment ({start_year}-{latest_year})")
    
    with tab2:
        # Calculate snapshot years (every 5 years)
        all_years = sorted(df_with_region["Year"].unique().to_list())
        snapshot_years = [year for year in all_years if year % 5 == 0][-3:]  # Last 3 snapshot years
        
        for end_year in snapshot_years:
            start_year = end_year - 4
            snapshot_data = (df_with_region
                .filter(pl.col("Year").is_between(start_year, end_year))
                .group_by(["ISO-code", "Economy", "Region"])
                .agg([
                    pl.col("DataValue").sum().alias("DataValue")
                ])
                .with_columns([
                    (pl.col("DataValue") / 1000).alias("DataValue_B")
                ])
            )
            
            create_choropleth(snapshot_data, f"5-Year Investment ({start_year}-{end_year})")

def create_choropleth(data, title):
    """Helper function to create choropleth map"""
    value_min = float(data["DataValue_B"].min())
    value_max = float(data["DataValue_B"].max())
    
    fig = go.Figure()
    fig.add_trace(go.Choropleth(
        locations=data["ISO-code"],
        z=data["DataValue_B"],
        text=data["Economy"],
        colorscale=[
            [0, '#f7fbff'],      # Very light blue
            [0.01, '#deebf7'],   # Light blue
            [0.02, '#c6dbef'],   # Pale blue
            [0.05, '#9ecae1'],   # Sky blue
            [0.1, '#6baed6'],    # Medium blue
            [0.2, '#4292c6'],    # Blue
            [0.3, '#2171b5'],    # Deep blue
            [0.5, '#08519c'],    # Dark blue
            [0.7, '#08306b'],    # Very dark blue
            [0.9, '#042144'],    # Navy
            [1.0, '#021b35']     # Deep navy
        ],
        zmin=0,
        zmax=max(1, value_max),
        marker_line_color='white',
        marker_line_width=0.5,
        colorbar=dict(
            title="Investment (USD Billion)",
            thickness=15,
            len=0.9,
            tickfont=dict(size=12, color='white'),
            titlefont=dict(size=14, color='white'),
            tickformat='.2f'
        ),
        hovertemplate="<b>%{text}</b><br>" +
                     "Investment: $%{z:.2f}B<br>" +
                     "<extra></extra>"
    ))
    
    fig.update_layout(
        title=title,
        geo=dict(
            showframe=False,
            showcoastlines=True,
            projection_type='equirectangular',
            bgcolor='rgba(0,0,0,0)'
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=600,
        margin=dict(l=0, r=0, t=30, b=0)
    )
    
    st.plotly_chart(fig, use_container_width=True)

def technology_mix(df, title):
    """Analyze and visualize the technology mix in investments"""
    st.subheader("Technology Distribution Analysis")
    
    # Group data by Technology and Year
    tech_data = (df
        .group_by(["Technology", "Year"])
        .agg([
            pl.col("DataValue").sum().alias("Investment")
        ])
        .filter(pl.col("Technology").is_not_null())
    ).to_pandas()
    
    if tech_data.empty:
        st.warning("No technology data available for the selected filters.")
        return
    
    # Sort technologies by total investment
    tech_totals = tech_data.groupby("Technology")["Investment"].sum().sort_values(ascending=False)
    available_techs = tech_totals.index.tolist()
    
    # Technology selector with default selection of top 3
    selected_techs = st.multiselect(
        "Select Technologies to Compare",
        options=available_techs,
        default=available_techs[:3],
        help="Select specific technologies to focus on their investment trends. Default shows top 3 by investment volume."
    )
    
    if not selected_techs:
        st.warning("Please select at least one technology to display.")
        return
    
    # Filter and sort data
    filtered_df = tech_data[tech_data["Technology"].isin(selected_techs)].copy()
    filtered_df["Year"] = filtered_df["Year"].astype(int)
    filtered_df = filtered_df.sort_values("Year")
    
    # Create stacked area chart
    fig = px.area(
        filtered_df,
        x="Year",
        y="Investment",
        color="Technology",
        title="Technology Investment Distribution Over Time",
        template="plotly_dark"
    )
    
    # Enhanced layout
    fig.update_layout(
        height=600,
        yaxis_title="Investment (USD Million)",
        hovermode="x unified",
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(17, 17, 17, 0.8)",
            bordercolor="rgba(255, 255, 255, 0.3)",
            borderwidth=1,
            font=dict(color="white")
        ),
        xaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor="rgba(255, 255, 255, 0.1)"
        ),
        yaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor="rgba(255, 255, 255, 0.1)"
        )
    )
    
    # Add hover template
    fig.update_traces(
        hovertemplate="<b>%{y:.1f}M USD</b><br>" +
                     "Year: %{x}<br>" +
                     "<extra></extra>"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Add insights section with improved formatting
    with st.expander("📊 Technology Investment Insights"):
        # Calculate insights
        total_by_tech = tech_data.groupby("Technology")["Investment"].sum()
        
        # Calculate growth rates
        growth_rates = {}
        for tech in selected_techs:
            tech_data_sorted = filtered_df[filtered_df["Technology"] == tech].sort_values("Year")
            if len(tech_data_sorted) >= 2:
                first_val = tech_data_sorted["Investment"].iloc[0]
                last_val = tech_data_sorted["Investment"].iloc[-1]
                if first_val > 0:
                    growth = ((last_val / first_val) - 1) * 100
                    growth_rates[tech] = growth
        
        # Sort technologies by investment
        selected_totals = total_by_tech[selected_techs].sort_values(ascending=False)
        
        st.markdown("### Key Findings")
        
        st.markdown("#### 🔹 Investment Overview")
        for tech, inv in selected_totals.items():
            st.markdown(f"- **{tech}**: ${inv/1000:.1f}B total investment")
        
        st.markdown("#### 🔹 Growth Trends")
        for tech, growth in sorted(growth_rates.items(), key=lambda x: x[1], reverse=True):
            st.markdown(f"- **{tech}**: {growth:.1f}% overall growth")
        
        st.markdown("#### 🔹 Latest Year Analysis")
        latest_year = filtered_df["Year"].max()
        latest_data = filtered_df[filtered_df["Year"] == latest_year]
        for tech in selected_techs:
            tech_inv = latest_data[latest_data["Technology"] == tech]["Investment"].iloc[0]
            st.markdown(f"- **{tech}**: ${tech_inv/1000:.1f}B in {latest_year}")

def top_performers(df, income_groups, title):
    """Analyze and visualize top performing countries by region"""
    st.subheader("Top Investment Leaders")
    
    # Merge with income groups
    df_with_income = df.join(
        income_groups,
        left_on="Economy",
        right_on="Economy",
        how="inner"
    )
    
    # Get unique regions and add "All Regions" option
    regions = ["All Regions"] + sorted(df_with_income["Region"].unique().to_list())
    
    # Region selector
    selected_region = st.selectbox(
        "Select Region",
        options=regions,
        help="Filter to see top performing countries within a specific region"
    )
    
    # Filter data based on region selection
    if selected_region != "All Regions":
        filtered_df = df_with_income.filter(pl.col("Region") == selected_region)
    else:
        filtered_df = df_with_income
    
    # Calculate total investment by country
    top_countries = (filtered_df
        .group_by(["Economy", "Income group"])
        .agg([
            pl.col("DataValue").sum().alias("Total_Investment"),
            pl.col("DataValue").mean().alias("Avg_Annual_Investment")
        ])
        .sort("Total_Investment", descending=True)
        .head(10)
        .to_pandas()
    )
    
    # Create bar chart
    fig = px.bar(
        top_countries,
        x="Economy",
        y="Total_Investment",
        color="Income group",
        title=f"Top 10 Countries {f'in {selected_region}' if selected_region != 'All Regions' else ''}",
        template="plotly_dark",
        hover_data=["Avg_Annual_Investment"]
    )
    
    fig.update_layout(
        height=500,
        xaxis_title="Country",
        yaxis_title="Total Investment (USD Million)",
        xaxis_tickangle=-45,
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor="rgba(0,0,0,0.8)",
            font_size=12
        )
    )
    
    fig.update_traces(
        hovertemplate="<b>%{x}</b><br>" +
                     "Total Investment: $%{y:.1f}M<br>" +
                     "Avg Annual: $%{customdata[0]:.1f}M<br>" +
                     "<extra></extra>"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Add insights with improved formatting
    with st.expander("📊 Investment Insights"):
        total_investment = top_countries["Total_Investment"].sum()
        avg_investment = top_countries["Avg_Annual_Investment"].mean()
        top_income_group = top_countries.groupby("Income group")["Total_Investment"].sum().idxmax()
        
        # Format income group distribution
        income_dist = top_countries['Income group'].value_counts()
        income_dist_text = "\n".join([f"  • {group}: {count} countries" 
                                    for group, count in income_dist.items()])
        
        st.markdown(f"""
        ### Key Findings {f'for {selected_region}' if selected_region != 'All Regions' else ''}
        
        💰 Investment Overview
        • Total investment in top 10: ${total_investment/1000:.1f}B
        • Average annual investment per country: ${avg_investment:.1f}M
        
        📊 Income Group Distribution
        • Dominant income group: {top_income_group}
        • Distribution:
        {income_dist_text}
        
        🏆 Top Performer Details
        • Leading country: {top_countries.iloc[0]['Economy']}
        • Investment: ${top_countries.iloc[0]['Total_Investment']/1000:.1f}B
        """)

def growth_rate_analysis(df, income_groups, title):
    """Analyze and visualize investment growth rates"""
    st.subheader("Investment Growth Patterns")
    
    # Add explanation of growth rate calculation
    with st.expander("ℹ️ How is Growth Rate Calculated?"):
        st.markdown("""
        **Growth Rate Calculation Method:**
        - Year-over-year percentage change in investment
        - Formula: (Current Year Investment - Previous Year Investment) / Previous Year Investment
        - Filtered to remove extreme outliers (>1000% growth) for better visualization
        """)
    
    # Merge with income groups
    df_with_income = df.join(
        income_groups,
        left_on="Economy",
        right_on="Economy",
        how="inner"
    )
    
    # Calculate year-over-year growth rates with outlier filtering
    growth_data = (df_with_income
        .sort("Year")
        .group_by(["Economy", "Region", "Year"])
        .agg([
            pl.col("DataValue").sum().alias("Investment")
        ])
        .sort(["Economy", "Year"])
    )
    
    growth_data = growth_data.with_columns([
        pl.col("Investment").pct_change().over(["Economy"]).alias("Growth_Rate")
    ]).filter(
        (pl.col("Growth_Rate").is_not_null()) &
        (pl.col("Growth_Rate").abs() <= 10)  # Filter extreme outliers
    )
    
    if growth_data.is_empty():
        st.warning("Insufficient data for growth rate analysis.")
        return
        
    # Convert to pandas for plotting
    growth_df = growth_data.to_pandas()
    
    # Create tabs for different views
    tab1, tab2 = st.tabs(["Regional Growth Trends", "Growth Distribution"])
    
    with tab1:
        # Calculate median growth rates by region and year for cleaner visualization
        median_growth = growth_df.groupby(['Year', 'Region'])['Growth_Rate'].median().reset_index()
        
        fig1 = px.line(
            median_growth,
            x="Year",
            y="Growth_Rate",
            color="Region",
            title="Median Growth Rate by Region",
            template="plotly_dark"
        )
        
        fig1.update_layout(
            height=500,
            yaxis_title="Growth Rate (%)",
            yaxis_tickformat=".0%",
            hovermode="x unified"
        )
        
        st.plotly_chart(fig1, use_container_width=True)
    
    with tab2:
        fig2 = px.box(
            growth_df,
            x="Region",
            y="Growth_Rate",
            color="Region",
            title="Growth Rate Distribution by Region",
            template="plotly_dark",
            points="outliers"
        )
        
        fig2.update_layout(
            height=500,
            yaxis_title="Growth Rate (%)",
            yaxis_tickformat=".0%",
            xaxis_tickangle=-45
        )
        
        st.plotly_chart(fig2, use_container_width=True)
    
    # Add key insights
    st.markdown("### Key Insights")
    
    # Calculate insights
    median_growth_by_region = growth_df.groupby('Region')['Growth_Rate'].agg(['median', 'std']).round(3)
    fastest_growing = median_growth_by_region.nlargest(1, 'median').index[0]
    most_stable = median_growth_by_region.nsmallest(1, 'std').index[0]
    
    st.markdown(f"""
    - **Fastest Growing Region**: {fastest_growing} shows the highest median growth rate
    - **Most Stable Growth**: {most_stable} demonstrates the most consistent growth pattern
    - **Growth Patterns**: Most regions show positive but volatile growth rates
    - **Investment Maturity**: Developed regions typically show more stable but lower growth rates
    """)

def investment_impact_analysis(df, income_groups, title):
    """Analyze and visualize investment impact"""
    st.subheader("Investment Impact Analysis")
    
    with st.expander("ℹ️ About This Analysis"):
        st.markdown("""
        **What This Shows:**
        - Annual investment trends by region
        - Cumulative investment growth (always increasing)
        - Regional investment patterns
        - Year-over-year changes
        """)
    
    # Merge with income groups and ensure data is properly sorted
    df_with_income = df.join(
        income_groups,
        left_on="Economy",
        right_on="Economy",
        how="inner"
    )
    
    # Calculate annual investments by region and year
    annual_data = (df_with_income
        .group_by(["Region", "Year"])
        .agg([
            pl.col("DataValue").sum().alias("Annual_Investment")
        ])
        .sort(["Region", "Year"])  # Ensure data is sorted for correct cumulative calc
    )
    
    # Convert to pandas and ensure proper sorting
    impact_df = annual_data.to_pandas()
    impact_df = impact_df.sort_values(['Region', 'Year'])
    
    # Calculate cumulative sum (will always increase)
    impact_df['Cumulative_Investment'] = impact_df.groupby('Region')['Annual_Investment'].cumsum()
    
    # Create tabs for different views
    tab1, tab2 = st.tabs(["Annual Trends", "Cumulative Growth"])
    
    with tab1:
        # Annual investment bar chart
        fig1 = px.bar(
            impact_df,
            x="Year",
            y="Annual_Investment",
            color="Region",
            title="Annual Investment by Region",
            template="plotly_dark",
            barmode="group"
        )
        
        fig1.update_layout(
            height=500,
            yaxis_title="Annual Investment (USD Million)",
            xaxis_tickangle=0,
            hovermode="x unified"
        )
        
        st.plotly_chart(fig1, use_container_width=True)
    
    with tab2:
        # Cumulative investment line chart (should always increase)
        fig2 = px.line(
            impact_df,
            x="Year",
            y="Cumulative_Investment",
            color="Region",
            title="Cumulative Investment Growth by Region",
            template="plotly_dark"
        )
        
        fig2.update_layout(
            height=500,
            yaxis_title="Cumulative Investment (USD Million)",
            xaxis_tickangle=0,
            hovermode="x unified"
        )
        
        st.plotly_chart(fig2, use_container_width=True)

def regional_analysis(df, income_groups, title):
    """Create a regional analysis visualization."""
    grouped_data = (df
        .group_by(["Year", "Region"])
        .agg(pl.col("DataValue").sum())
        .sort("Year")
    )
    
    fig = px.line(
        grouped_data.to_pandas(), 
        x="Year", 
        y="DataValue", 
        color="Region",
        title=f"{title} by Region"
    )
    
    fig.update_layout(
        yaxis_title="Investment (2020 USD million)",
        xaxis_title="Year",
        height=600,
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.02
        )
    )
    
    return fig

def investment_trends(df, income_groups):
    """Analyze investment trends over time."""
    global_trends = (df
        .group_by("Year")
        .agg(pl.col("DataValue").sum())
        .sort("Year")
    )
    
    fig = px.line(
        global_trends.to_pandas(),
        x="Year",
        y="DataValue",
        title="Global Investment Trends"
    )
    
    fig.update_layout(
        yaxis_title="Investment (2020 USD million)",
        xaxis_title="Year",
        height=600
    )
    
    return fig

def roi_analysis(df, income_groups, window_years=5):
    """Calculate and visualize ROI trends"""
    st.subheader("Return on Investment Analysis")
    
    with st.expander("ℹ️ How is ROI Calculated?"):
        st.markdown("""
        **ROI Calculation Method:**
        1. Calculate year-over-year investment growth rate
        2. Use a rolling window to smooth out volatility
        3. Filter out extreme outliers for better visualization
        4. Show trends in investment returns over time
        """)
    
    # Merge data and ensure proper sorting
    df_with_income = df.join(
        income_groups,
        left_on="Economy",
        right_on="Economy",
        how="inner"
    )
    
    # Calculate annual investments by region and year
    annual_data = (df_with_income
        .group_by(["Region", "Year"])
        .agg([
            pl.col("DataValue").sum().alias("Investment")
        ])
        .sort(["Region", "Year"])
    )
    
    # Convert to pandas for calculations
    roi_df = annual_data.to_pandas()
    
    # Calculate growth rates for each region separately
    regions = roi_df['Region'].unique()
    all_roi_data = []
    
    for region in regions:
        region_data = roi_df[roi_df['Region'] == region].copy()
        region_data = region_data.sort_values('Year')
        
        # Calculate growth rate
        region_data['Growth_Rate'] = region_data['Investment'].pct_change()
        
        # Calculate rolling ROI
        region_data['Rolling_ROI'] = region_data['Growth_Rate'].rolling(
            window=window_years, 
            min_periods=1
        ).mean()
        
        all_roi_data.append(region_data)
    
    # Combine all regions back together
    roi_df = pd.concat(all_roi_data, ignore_index=True)
    
    # Filter out extreme values for better visualization
    roi_df = roi_df[
        (roi_df['Rolling_ROI'].notna()) &
        (roi_df['Rolling_ROI'] > -1) & 
        (roi_df['Rolling_ROI'] < 2)  # Filter extreme outliers
    ]
    
    # Create visualization tabs
    tab1, tab2 = st.tabs(["ROI Trends", "Annual Growth Rates"])
    
    with tab1:
        # Rolling ROI trends
        fig1 = px.line(
            roi_df,
            x="Year",
            y="Rolling_ROI",
            color="Region",
            title=f"{window_years}-Year Rolling ROI by Region",
            template="plotly_dark"
        )
        
        fig1.update_layout(
            height=500,
            yaxis_title="Rolling ROI (%)",
            yaxis_tickformat=".0%",
            hovermode="x unified"
        )
        
        st.plotly_chart(fig1, use_container_width=True)
    
    with tab2:
        # Annual growth rates
        fig2 = px.bar(
            roi_df,
            x="Year",
            y="Growth_Rate",
            color="Region",
            title="Annual Investment Growth Rates",
            template="plotly_dark",
            barmode="group"
        )
        
        fig2.update_layout(
            height=500,
            yaxis_title="Growth Rate (%)",
            yaxis_tickformat=".0%",
            xaxis_tickangle=0
        )
        
        st.plotly_chart(fig2, use_container_width=True)
    
    # Add summary statistics
    st.markdown("### Regional ROI Summary")
    
    summary_stats = roi_df.groupby('Region').agg({
        'Rolling_ROI': ['mean', 'std', 'min', 'max']
    }).round(3)
    
    summary_stats.columns = ['Average ROI', 'ROI Volatility', 'Minimum ROI', 'Maximum ROI']
    summary_stats = summary_stats.sort_values('Average ROI', ascending=False)
    
    st.dataframe(
        summary_stats,
        column_config={
            "Average ROI": st.column_config.NumberColumn(
                "Average ROI",
                format="%.1%"
            ),
            "ROI Volatility": st.column_config.NumberColumn(
                "Volatility",
                format="%.1%"
            ),
            "Minimum ROI": st.column_config.NumberColumn(
                "Min ROI",
                format="%.1%"
            ),
            "Maximum ROI": st.column_config.NumberColumn(
                "Max ROI",
                format="%.1%"
            )
        }
    )
    
    # Add insights
    st.markdown("### Key Insights")
    
    # Calculate insights
    best_roi_region = summary_stats.index[0]
    most_stable_region = summary_stats.sort_values('ROI Volatility').index[0]
    
    st.markdown(f"""
    #### Performance Analysis
    - **Highest Returns**: {best_roi_region} shows the strongest average ROI
    - **Most Stable**: {most_stable_region} demonstrates the most consistent returns
    
    #### Investment Patterns
    - ROI tends to be more volatile in developing regions
    - Established markets show more stable but generally lower returns
    - Growth rates vary significantly across regions and time periods
    """)

def setup_sidebar_filters(df):
    """Setup sidebar filters for the application"""
    st.sidebar.header("Global Filters")
    
    # Year range selector
    year_range = st.sidebar.slider(
        "Select Year Range",
        min_value=int(df["Year"].min()),
        max_value=int(df["Year"].max()),
        value=(int(df["Year"].min()), int(df["Year"].max()))
    )
    
    # Region multiselect
    all_regions = df["Region"].unique().to_list()
    selected_regions = st.sidebar.multiselect(
        "Select Regions",
        options=all_regions,
        default=all_regions
    )
    
    return year_range, selected_regions

def income_based_analysis(df, income_groups, title):
    """Enhanced visualization of income-based investment patterns with advanced features"""
    
    # Merge investment data with income groups
    df_with_income = df.join(
        income_groups,
        left_on="Economy",
        right_on="Economy",
        how="inner"
    )
    
    # Add tabs for different analysis views
    tab1, tab2, tab3 = st.tabs(["Investment Trends", "Regional Distribution", "Growth Analysis"])
    
    with tab1:
        # Enhanced time series visualization
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Interactive time series
            income_trends = (df_with_income
                .group_by(["Year", "Income group"])
                .agg(pl.col("DataValue").sum().alias("Investment"))
                .sort("Year")
            ).to_pandas()
            
            # Custom color palette
            income_colors = {
                'High income': '#2ecc71',
                'Upper middle income': '#3498db',
                'Lower middle income': '#9b59b6',
                'Low income': '#e74c3c'
            }
            
            # Add visualization type selector
            viz_type = st.selectbox(
                "Select Visualization Type",
                ["Area", "Line", "Bar"],
                key="income_viz_type"
            )
            
            if viz_type == "Area":
                fig1 = px.area(
                    income_trends,
                    x="Year",
                    y="Investment",
                    color="Income group",
                    color_discrete_map=income_colors,
                    title="Investment Trends by Income Group",
                    template="plotly_dark"
                )
            elif viz_type == "Line":
                fig1 = px.line(
                    income_trends,
                    x="Year",
                    y="Investment",
                    color="Income group",
                    color_discrete_map=income_colors,
                    title="Investment Trends by Income Group",
                    template="plotly_dark"
                )
            else:  # Bar
                fig1 = px.bar(
                    income_trends,
                    x="Year",
                    y="Investment",
                    color="Income group",
                    color_discrete_map=income_colors,
                    title="Investment Trends by Income Group",
                    template="plotly_dark"
                )
            
            fig1.update_layout(
                height=500,
                hovermode="x unified",
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01,
                    bgcolor='rgba(17, 17, 17, 0.8)',
                    bordercolor="rgba(255, 255, 255, 0.3)",
                    borderwidth=1,
                    font=dict(color='white')
                )
            )
            
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            # Add year selector for distribution
            latest_year = df_with_income["Year"].max()
            
            # Distribution for selected year
            year_data = (df_with_income
                .filter(pl.col("Year") == latest_year)
                .group_by("Income group")
                .agg(pl.col("DataValue").sum().alias("Investment"))
            ).to_pandas()
            
            fig2 = px.pie(
                year_data,
                values="Investment",
                names="Income group",
                title=f"Investment Distribution ({latest_year})",
                color_discrete_map=income_colors,
                template="plotly_dark"
            )
            
            fig2.update_traces(
                textposition='inside',
                textinfo='percent+label'
            )
            
            st.plotly_chart(fig2, use_container_width=True)
    
    with tab2:
        # Regional distribution by income group
        st.subheader("Regional Distribution by Income Group")
        
        region_income = (df_with_income
            .group_by(["Region", "Income group"])
            .agg(pl.col("DataValue").sum().alias("Investment"))
            .sort("Investment", descending=True)
        ).to_pandas()
        
        fig3 = px.sunburst(
            region_income,
            path=['Region', 'Income group'],
            values='Investment',
            color='Income group',
            color_discrete_map=income_colors,
            template="plotly_dark"
        )
        
        fig3.update_layout(height=600)
        st.plotly_chart(fig3, use_container_width=True)
    
    with tab3:
        # Growth analysis
        st.subheader("Investment Growth Analysis")
        
        # Calculate growth rates
        growth_data = (df_with_income
            .group_by(["Year", "Income group"])
            .agg(pl.col("DataValue").sum().alias("Investment"))
            .sort("Year")
        ).to_pandas()
        
        # Add growth rate calculation
        growth_data['Growth'] = growth_data.groupby('Income group')['Investment'].pct_change() * 100
        
        fig4 = px.line(
            growth_data.dropna(),
            x="Year",
            y="Growth",
            color="Income group",
            color_discrete_map=income_colors,
            title="Year-over-Year Investment Growth Rate",
            template="plotly_dark"
        )
        
        fig4.update_layout(
            yaxis_title="Growth Rate (%)",
            height=500
        )
        
        st.plotly_chart(fig4, use_container_width=True)
    
    # Enhanced metrics dashboard
    st.markdown("### Investment Metrics Dashboard")
    
    # Create three rows of metrics
    row1_cols = st.columns(4)
    row2_cols = st.columns(4)
    row3_cols = st.columns(4)
    
    # Row 1: Total investments
    with row1_cols[0]:
        total_investment = df_with_income["DataValue"].sum()
        st.metric(
            "Total Investment",
            f"${total_investment/1000:.1f}B"
        )
    
    with row1_cols[1]:
        avg_investment = df_with_income["DataValue"].mean()
        st.metric(
            "Average Annual Investment",
            f"${avg_investment:.1f}M"
        )
    
    with row1_cols[2]:
        high_income_share = (
            year_data[year_data["Income group"] == "High income"]["Investment"].iloc[0] /
            year_data["Investment"].sum() * 100
        )
        st.metric(
            "High Income Share",
            f"{high_income_share:.1f}%"
        )
    
    with row1_cols[3]:
        low_income_share = (
            year_data[year_data["Income group"] == "Low income"]["Investment"].iloc[0] /
            year_data["Investment"].sum() * 100
        )
        st.metric(
            "Low Income Share",
            f"{low_income_share:.1f}%"
        )
    
    # Row 2: Growth metrics
    growth_metrics = growth_data.groupby('Income group')['Growth'].agg(['mean', 'max', 'min']).round(1)
    
    for i, income_group in enumerate(growth_metrics.index):
        with row2_cols[i]:
            st.metric(
                f"{income_group} Growth",
                f"{growth_metrics.loc[income_group, 'mean']:.1f}%",
                f"Max: {growth_metrics.loc[income_group, 'max']:.1f}%"
            )
    
    # Add trend analysis with expandable sections
    st.markdown("### Detailed Analysis")
    
    with st.expander("Investment Patterns"):
        st.markdown("""
        #### Key Findings
        - **High Income Countries**: Lead in investment volume with consistent growth
        - **Middle Income Transition**: Upper middle income countries show rapid growth
        - **Investment Gap**: Significant disparity between high and low income groups
        - **Growth Potential**: Lower middle income countries show promising growth rates
        """)
    
    with st.expander("Regional Insights"):
        st.markdown("""
        #### Regional Distribution
        - Analysis of investment patterns across regions
        - Identification of regional leaders and laggards
        - Assessment of regional growth potential
        """)
    
    with st.expander("Policy Implications"):
        st.markdown("""
        #### Policy Recommendations
        - Strategies for reducing investment gaps
        - Opportunities for cross-regional collaboration
        - Focus areas for development support
        """)

def main():
    """Main function for the Clean Energy Investment Explorer application"""
    
    # Page config with custom title and icon
    st.set_page_config(
        page_title="Clean Energy Investment Explorer",
        page_icon="🌍",
        layout="wide"
    )
    
    # Main title and introduction
    st.title("Clean Energy Investment Explorer")
    st.markdown("""
    ### Tracking Global Progress in Sustainable Energy Adoption
    
    This interactive dashboard explores the global transition to clean energy through the lens of 
    investment patterns, technological breakthroughs, and regional adoption rates. Aligned with the 
    UN's Sustainable Development Goal 7 (SDG7), it provides insights into the worldwide journey 
    toward sustainable and modern energy for all.
    
    #### 🎯 Key Focus Areas
    • Regional investment patterns and disparities
    • Technology adoption across income groups
    • Impact of technological breakthroughs
    • Progress tracking toward SDG7 goals
    
    #### 📊 How to Use This Dashboard
    Use the sidebar filters to focus on specific regions or time periods. Each section provides 
    different insights into the clean energy transition, from global trends to detailed country-level 
    analysis.
    """)
    
    # Add divider
    st.markdown("---")
    
    # Load data
    public_investment, income_groups = load_data()
    
    # Setup global filters in sidebar
    st.sidebar.markdown("## 🔍 Global Filters")
    selected_years, selected_regions = setup_sidebar_filters(public_investment)
    
    # Add context about the data in sidebar
    st.sidebar.markdown("""
    #### About the Data
    This dashboard uses public investment data in clean energy technologies, 
    tracking progress toward sustainable energy adoption globally.
    
    Data is categorized by:
    • Geographic regions
    • Income groups
    • Technology types
    • Investment volumes
    """)
    
    # Filter data based on global selections
    filtered_data = public_investment.filter(
        (pl.col("Year").is_between(selected_years[0], selected_years[1])) &
        (pl.col("Region").is_in(selected_regions))
    )
    
    # Display visualizations with enhanced section headers
    st.header("🌍 1. Global Investment Landscape")
    global_progress_visualization(filtered_data, "Clean Energy Investment")
    
    st.header("💰 2. Investment Distribution by Income Level")
    income_based_analysis(filtered_data, income_groups, "Clean Energy Investment")
    
    st.header("🗺️ 3. Regional Investment Patterns")
    regional_analysis_choropleth(filtered_data, income_groups, "Clean Energy Investment")
    
    st.header("⚡ 4. Technology Innovation & Adoption")
    technology_mix(filtered_data, "Clean Energy Investment")
    
    st.header("🏆 5. Leading Countries")
    top_performers(filtered_data, income_groups, "Clean Energy Investment")
    
    st.header("📈 6. Growth & Impact Analysis")
    growth_rate_analysis(filtered_data, income_groups, "Clean Energy Investment")
    
    # Add footer with additional context
    st.markdown("---")
    st.markdown("""
    #### About This Project
    This interactive visualization tool is part of a research project exploring clean energy adoption 
    and technological breakthroughs. It aims to provide stakeholders with insights for informed 
    decision-making in the global energy transition.
    
    *Data sources: Public investment data in renewable energy technologies, World Bank income classifications*
    """)

if __name__ == "__main__":
    main()