import os
import json
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from self_refine import TASKS, log, run_task, gpt4_pref_evaluation, parse_gpt4_pref

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def run_ablation_for_task(task_key, x, task_config, max_iter=3):
    configs = {
        "full": {"label": "Full Self-Refine", "use_feedback": True, "use_history": True},
        "no_feedback": {"label": "Sin feedback", "use_feedback": False, "use_history": True},
        "no_history": {"label": "Sin historial", "use_feedback": True, "use_history": False},
        "no_iterative": {"label": "Sin iterative (1 iter)", "use_feedback": True, "use_history": True, "max_iter": 1},
    }

    results = {}
    for cfg_key, cfg in configs.items():
        log("ABLATION", "RUNNING", f"{task_key} / {cfg_key}")

        cfg_max_iter = cfg.get("max_iter", max_iter)
        from self_refine import self_refine

        y_0, y_final = self_refine(
            x, task_config,
            max_iter=cfg_max_iter,
            use_feedback=cfg["use_feedback"],
            use_history=cfg["use_history"]
        )

        veredicto = gpt4_pref_evaluation(y_0, y_final, task_name=task_config["name"])
        winner = parse_gpt4_pref(veredicto)

        results[cfg_key] = {
            "label": cfg["label"],
            "winner": winner,
            "y_0_len": len(y_0),
            "y_final_len": len(y_final),
        }
        log("ABLATION", "RESULT", f"{task_key} / {cfg_key}: winner={winner}")

    return results


def run_all_ablations(max_iter=3):
    all_results = {}

    for task_key in TASKS:
        task_config = TASKS[task_key]
        x = task_config["examples"][0]
        log("ABLATION", "TASK_START", f"\n{'='*60}\n  TAREA: {task_key}\n{'='*60}")

        task_results = run_ablation_for_task(task_key, x, task_config, max_iter=max_iter)
        all_results[task_key] = task_results

    report_path = os.path.join(RESULTS_DIR, "ablation_self_refine_report.json")
    with open(report_path, "w") as f:
        json.dump(all_results, f, indent=2)

    log("ABLATION", "DONE", f"Reporte guardado en {report_path}")
    print("\n" + "=" * 70)
    print(" RESUMEN DE ABLATION STUDIES — SELF-REFINE")
    print("=" * 70)
    print(f"{'Tarea'.ljust(25)} {'Full'.ljust(15)} {'No Fdbk'.ljust(15)} {'No Hist'.ljust(15)} {'1 Iter'.ljust(15)}")
    print("-" * 70)
    for task_key, res in all_results.items():
        full = res.get("full", {}).get("winner", "-")
        no_fb = res.get("no_feedback", {}).get("winner", "-")
        no_hist = res.get("no_history", {}).get("winner", "-")
        no_iter = res.get("no_iterative", {}).get("winner", "-")
        print(f"{task_key.ljust(25)} {full.ljust(15)} {no_fb.ljust(15)} {no_hist.ljust(15)} {no_iter.ljust(15)}")
    print("=" * 70)

    return all_results


if __name__ == "__main__":
    max_iter = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    run_all_ablations(max_iter=max_iter)
