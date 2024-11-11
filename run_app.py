import streamlit.web.cli as stcli
import sys
from pathlib import Path

if __name__ == "__main__":
    app_path = Path(__file__).parent / "src" / "sdg7_explorer" / "app.py"
    sys.argv = ["streamlit", "run", str(app_path)]
    sys.exit(stcli.main())