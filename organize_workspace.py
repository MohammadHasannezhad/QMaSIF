"""
Safe, non-destructive workspace organizer for QMaSIF before pushing to GitHub.

What it does:
  1. Renames 'tep1_verify_core.py' -> 'step1_verify_core.py' (if present).
  2. Copies publication figures ('Fig*.png') into 'figures/' cleanly.
  3. Copies saved benchmark outputs ('step*.json') into 'results/' cleanly
     (keeps root 'step7_cluster_cv_results.json' intact so step8/step9 never break!).
  4. Copies PyMOL visualization files ('1AK4_qmasif_annotated.pdb', 'visualize_1ak4_qmasif.pml')
     into 'pymol/' so '@visualize_1ak4_qmasif.pml' works out-of-the-box.
  5. Audits all expected core files and prints a ready-to-push summary.

Usage:
    python organize_workspace.py --dry-run   # Preview actions without changing anything
    python organize_workspace.py --apply     # Apply organization safely
"""

import argparse
import glob
import os
import shutil


PYMOL_FILES = [
    "1AK4_qmasif_annotated.pdb",
    "visualize_1ak4_qmasif.pml",
]

REQUIRED_ROOT_SCRIPTS = [
    "qmasif_core.py",
    "step1_verify_core.py",
    "step2_build_pdb_dataset.py",
    "step3_train_and_benchmark.py",
    "step4_ablations.py",
    "step5_nisq_and_bp.py",
    "step6_pymol_interpretability.py",
    "step7_priority1_cluster_cv.py",
    "step8_priority2_3_ablation_and_symmetry.py",
    "step9_final_reviewer_checks.py",
    "update_fig2_for_overleaf.py",
]


def main():
    parser = argparse.ArgumentParser(description="Organize QMaSIF workspace cleanly for GitHub.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="Show planned actions without modifying files")
    mode.add_argument("--apply", action="store_true", help="Apply safe organization")
    args = parser.parse_args()

    dry = args.dry_run
    tag = "[DRY-RUN]" if dry else "[APPLY]"

    print("=" * 80)
    print(f"QMaSIF WORKSPACE ORGANIZER ({'PREVIEW ONLY' if dry else 'APPLYING CHANGES'})")
    print("=" * 80)

    for d in ["figures", "results", "pymol", "docs"]:
        if not os.path.isdir(d):
            print(f"  {tag} Create directory: {d}/")
            if not dry:
                os.makedirs(d, exist_ok=True)

    # 1. Fix typo in step1 filename if present
    if os.path.exists("tep1_verify_core.py") and not os.path.exists("step1_verify_core.py"):
        print(f"  {tag} Rename: tep1_verify_core.py -> step1_verify_core.py")
        if not dry:
            os.rename("tep1_verify_core.py", "step1_verify_core.py")

    # 2. Organize PNG figures into figures/
    pngs = sorted(glob.glob("Fig*.png"))
    for p in pngs:
        dst = os.path.join("figures", os.path.basename(p))
        print(f"  {tag} Copy figure: {p} -> {dst}")
        if not dry:
            shutil.copy2(p, dst)

    # 3. Archive JSON results into results/ (keep root JSONs so step8/step9 dependencies stay intact)
    jsons = sorted(glob.glob("step*.json"))
    for j in jsons:
        dst = os.path.join("results", os.path.basename(j))
        print(f"  {tag} Archive JSON: {j} -> {dst}")
        if not dry:
            shutil.copy2(j, dst)

    # 4. Copy PyMOL artifacts into pymol/
    for pf in PYMOL_FILES:
        if os.path.exists(pf):
            dst = os.path.join("pymol", pf)
            print(f"  {tag} Copy PyMOL file: {pf} -> {dst}")
            if not dry:
                shutil.copy2(pf, dst)

    # 5. Pre-push audit
    print("\n--- Pre-Push Checklist ---")
    for fname in REQUIRED_ROOT_SCRIPTS + ["qmasif_best_model.pt"]:
        exists = os.path.exists(fname) or (fname == "step1_verify_core.py" and os.path.exists("tep1_verify_core.py"))
        print(f"  [{'OK' if exists else 'MISSING':>7}] {fname}")

    if dry:
        print("\nPreview complete. Run 'python organize_workspace.py --apply' to apply changes.")
    else:
        print("\nOrganization complete! Run 'python run_all.py --check' before 'git push'.")


if __name__ == "__main__":
    main()
