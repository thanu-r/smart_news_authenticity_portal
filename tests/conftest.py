import os
import sys
from pathlib import Path

# Add project root and ml-service to python path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "ml-service"))

# Preload torch and add DLL directory on Windows
try:
    torch_lib = project_root / "venv" / "Lib" / "site-packages" / "torch" / "lib"
    if torch_lib.exists() and hasattr(os, "add_dll_directory"):
        os.add_dll_directory(str(torch_lib))
    import torch
except Exception as e:
    print(f"Torch preload warning: {e}")
