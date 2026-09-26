"""End-to-end runner: python run_pipeline.py --phase 1 | 2 | 3 | 4 | 5 | all"""
import argparse
import sys
import warnings

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="all", choices=["1", "2", "3", "4", "5", "all"])
    phase = ap.parse_args().phase
    if phase in ("1", "all"):
        from src.phase1_merge import run_phase1
        run_phase1()
    if phase in ("2", "all"):
        from src.phase2_eda import run_phase2
        run_phase2()
    if phase in ("3", "all"):
        from src.phase3_models import run_phase3
        run_phase3()
    if phase in ("4", "all"):
        from src.phase4_combination import run_phase4
        run_phase4()
    if phase in ("5", "all"):
        from src.phase5_survival import run_phase5
        run_phase5()


if __name__ == "__main__":
    main()
