import sys
from pathlib import Path

# Add the repository root to the python path so that IDEs and pytest
# can correctly resolve imports like `from scripts.run_pipeline import main`
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
