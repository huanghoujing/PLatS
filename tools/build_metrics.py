#!/usr/bin/env python3
"""Build both exact topology modules locally; no downloads or system installs."""
import os,shutil,subprocess,sys
from pathlib import Path
import pybind11
root=Path(__file__).resolve().parents[1]
cmake=shutil.which('cmake') or str(Path(sys.executable).parent/'cmake')
for relative in ['topometrics/external/Betti-Matching-3D','betti_compact']:
 source=root/'third_party'/relative
 subprocess.run([cmake,'-S',str(source),'-B',str(source/'build'),
   '-Dpybind11_DIR='+pybind11.get_cmake_dir(),'-DPYBIND11_FINDPYTHON=ON',
   '-DPython_EXECUTABLE='+sys.executable,'-DCMAKE_BUILD_TYPE=Release',
   '-DCMAKE_POLICY_VERSION_MINIMUM=3.5'],check=True)
 subprocess.run([cmake,'--build',str(source/'build'),'--parallel',os.environ.get('BUILD_JOBS','8')],check=True)
print('Exact topology modules built in this bundle.')
