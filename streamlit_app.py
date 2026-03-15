import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "D603"))

from streamlit_app_task3 import main

if __name__ == "__main__":
    main()
