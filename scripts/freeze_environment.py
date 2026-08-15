#!/usr/bin/env python3
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))

import subprocess
from pathlib import Path
root=Path(__file__).resolve().parents[1]
out=root/"results"/"logs"/"pip_freeze_bt.txt"
out.parent.mkdir(parents=True,exist_ok=True)
txt=subprocess.check_output([sys.executable,"-m","pip","freeze"],text=True)
out.write_text(txt,encoding="utf-8")
print(out)
