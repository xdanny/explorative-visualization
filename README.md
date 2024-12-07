## Clean Energy Investment Explorer
An interactive dashboard for exploring global clean energy adoption patterns, technological breakthroughs, and investment trends. Built with Streamlit and powered by IRENA's public investment data. The app is currently hosted on StreamLit Cloud and can be accessed [here](https://explorative-visualization-rmmwdmd3dsjnbwkso9d9ov.streamlit.app/).

### 🎯 Project Overview
This visualization tool provides insights into:
- Global investment landscapes
- Regional adoption patterns
- Technology distribution
- Income-based analysis
- Country-specific trends

### 🛠️ Prerequisites
- Python 3.8 or higher
- uv (Python package installer)
- Git   

### 📦 Installation
- Windows
```
# Clone the repository
git clone https://github.com/yourusername/clean-energy-explorer.git
cd clean-energy-explorer

# Create and activate virtual environment using uv
uv venv
.venv\Scripts\activate

# Install dependencies
uv pip install -r requirements.txt
```

- MacOS/Linux
```
# Clone the repository
git clone https://github.com/yourusername/clean-energy-explorer.git
cd clean-energy-explorer

# Create and activate virtual environment using uv
uv venv
source .venv/bin/activate

# Install dependencies
uv pip install -r requirements.txt
```

### 🚀 Running the Application
```
# Run the Streamlit app
python ./run_app.py
```

### 📊 Dependencies
- streamlit
- polars
plotly
- altair
- pandas


### 🔧 Troubleshooting
If you encounter any issues:
- Ensure your Python version is compatible:
- Bash
- Verify virtual environment activation:
- Windows: You should see (.venv) in your terminal
macOS/Linux: Run which python to confirm it points to your virtual environment

### 📝 Data Sources
This project uses data from:
- [IRENA's Investment Data](https://www.irena.org/Energy-Transition/Finance-and-investment/Investment)
- [World Bank income classifications](https://data.worldbank.org/indicator/NY.GDP.PCAP.CD)