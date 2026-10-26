
# ==========================================================
# Ficheiro principal da aplicação Android
# ==========================================================

import runpy
from pathlib import Path

ficheiro_principal = Path(__file__).resolve().parent / "oraculoGui.py"

if not ficheiro_principal.is_file():
    raise FileNotFoundError(f"Ficheiro principal não encontrado: {ficheiro_principal}")

runpy.run_path(str(ficheiro_principal), run_name="__main__")
