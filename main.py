"""
Main entry point for the CMMV portfolio replication and extension.

Runs the full pipeline:
  1. Replication  — cost-free Sharpe ratio table (Table 1)
  2. Replication figures — Figures 1-3
  3. Extension   — transaction costs + sensitivity analysis
  4. Extension figures — Figures 4-8

All outputs saved to /results/ (CSV) and /figures/ (PNG).

Usage:
    python main.py
"""

import time
import os
import sys

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    t_total = time.time()

    print("=" * 72)
    print("CMMV PORTFOLIO — FULL PIPELINE")
    print("Wang, Jin, Wu & Gao (2026), Automatica 183, 112669")
    print("=" * 72)

    # ------------------------------------------------------------------
    # Step 1: Replication
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("STEP 1/4: Running replication (cost-free SR table, q=1..10)")
    print("=" * 72)
    from replication.run_replication import main as run_replication
    run_replication()

    # ------------------------------------------------------------------
    # Step 2: Replication figures
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("STEP 2/4: Generating replication figures (Figures 1-3)")
    print("=" * 72)
    from replication.generate_figures import figure1, figure2, figure3, comparison_table
    figure1()
    figure2()
    figure3()
    comparison_table()

    # ------------------------------------------------------------------
    # Step 3: Extension
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("STEP 3/4: Running extension (transaction costs + sensitivity)")
    print("=" * 72)
    from extension.run_extension import main as run_extension
    run_extension()

    # ------------------------------------------------------------------
    # Step 4: Extension figures
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("STEP 4/4: Generating extension figures (Figures 4-8)")
    print("=" * 72)
    from extension.generate_figures import figure4, figure5, figure6, figure7, figure8
    figure4()
    figure5()
    figure6()
    figure7()
    figure8()

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    elapsed = time.time() - t_total
    print("\n" + "=" * 72)
    print(f"ALL DONE — Total time: {elapsed:.1f}s")
    print("=" * 72)

    results_dir = os.path.join(os.path.dirname(__file__), 'results')
    figures_dir = os.path.join(os.path.dirname(__file__), 'figures')

    csv_files = [f for f in os.listdir(results_dir) if f.endswith('.csv')]
    png_files = [f for f in os.listdir(figures_dir) if f.endswith('.png')]

    print(f"\nResults: {len(csv_files)} CSV files in /results/")
    for f in sorted(csv_files):
        print(f"  - {f}")

    print(f"\nFigures: {len(png_files)} PNG files in /figures/")
    for f in sorted(png_files):
        print(f"  - {f}")

    print("\nAll outputs saved to /figures/ and /results/")


if __name__ == '__main__':
    main()
