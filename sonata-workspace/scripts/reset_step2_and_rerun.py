#!/usr/bin/env python3
"""
Удаляет частичные результаты шага 2 (Depth Pro) для указанных последовательностей
и запускает полный пайплайн. Шаг 2 будет выполнен заново только для этих seq.

Использование:
  export dataset=/workspace/dataset OUT=/workspace/dataset
  python scripts/reset_step2_and_rerun.py 00 01
  # или без аргументов — сбросить 00 и 01 по умолчанию и запустить пайплайн
"""
import os
import sys
import subprocess

def main():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(repo_root)
    out = os.environ.get("OUT", os.environ.get("dataset", "/workspace/dataset"))
    depth_base = os.path.join(out, "VoxFormerDepthPro", "depth", "sequences")
    if not os.path.isdir(depth_base):
        print("No depth dir at", depth_base)
        sys.exit(1)
    seqs = sys.argv[1:] if len(sys.argv) > 1 else ["00", "01"]
    for seq in seqs:
        d = os.path.join(depth_base, seq)
        if os.path.isdir(d):
            import shutil
            shutil.rmtree(d)
            print("Removed", d)
        else:
            print("No dir", d, "(skip)")
    print("Running full pipeline...")
    subprocess.run([sys.executable, "scripts/run_depthpro_to_sonata.py"], cwd=repo_root, check=True)

if __name__ == "__main__":
    main()
