import os
import json
import sys
import time
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from human_eval_runner import run_humaneval_problem
from reflexion import log
from human_eval.data import read_problems, write_jsonl
from human_eval.evaluation import evaluate_functional_correctness

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def run_ablation(label, mode, max_intentos=4, sample_size=2):
    log("ABLATION", "START", f"{label}: mode={mode}, sample={sample_size} problemas")

    problems = read_problems()
    all_ids = sorted(problems.keys(), key=lambda x: int(x.split("/")[1]))[:sample_size]

    use_self_generated_tests = mode == "full" or mode == "no_memory" or mode == "no_reflection"
    use_reflection = mode == "full" or mode == "no_memory" or mode == "no_tests"
    use_memory = mode == "full" or mode == "no_tests" or mode == "no_reflection"

    if mode == "full":
        use_self_generated_tests = True
        use_reflection = True
        use_memory = True
    elif mode == "no_tests":
        use_self_generated_tests = False
        use_reflection = True
        use_memory = True
    elif mode == "no_reflection":
        use_self_generated_tests = True
        use_reflection = False
        use_memory = True
    elif mode == "no_memory":
        use_self_generated_tests = True
        use_reflection = True
        use_memory = False

    output_path = os.path.join(RESULTS_DIR, f"ablation_{label}_samples.jsonl")
    jsonl_data = []
    solved = 0

    for idx, task_id in enumerate(all_ids):
        problem = problems[task_id]
        log("ABLATION", "PROGRESS", f"[{idx+1}/{len(all_ids)}] {task_id}")

        tarea = {
            "id": problem["task_id"],
            "descripcion": problem["prompt"],
            "tests": None,
        }

        from reflexion import ejecutar_agente_reflexion
        try:
            exito, codigo_final = ejecutar_agente_reflexion(
                tarea,
                max_intentos=max_intentos,
                use_self_generated_tests=use_self_generated_tests,
                use_reflection=use_reflection,
                use_memory=use_memory
            )

            from human_eval_runner import extract_body_from_completion
            body = extract_body_from_completion(codigo_final, problem["entry_point"])

            jsonl_data.append({"task_id": task_id, "completion": body})
            if exito:
                solved += 1
        except Exception as e:
            log("ABLATION", "ERROR", f"{task_id}: {str(e)}")
            jsonl_data.append({"task_id": task_id, "completion": ""})

    write_jsonl(output_path, jsonl_data)
    log("ABLATION", "EVAL", "Evaluando...")
    eval_results = evaluate_functional_correctness(
        sample_file=output_path,
        k=[1, 10],
        n_workers=4,
        timeout=3.0
    )

    summary = {
        "label": label,
        "mode": mode,
        "use_self_generated_tests": use_self_generated_tests,
        "use_reflection": use_reflection,
        "use_memory": use_memory,
        "total_problems": len(all_ids),
        "solved_in_loop": solved,
        "pass_at_1": eval_results.get("pass@1", 0),
        "pass_at_10": eval_results.get("pass@10", 0),
        "timestamp": datetime.now().isoformat(),
    }

    results_path = os.path.join(RESULTS_DIR, f"ablation_{label}.json")
    with open(results_path, "w") as f:
        json.dump(summary, f, indent=2)

    log("ABLATION", "RESULT", json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    sample_size = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    max_intentos = int(sys.argv[2]) if len(sys.argv) > 2 else 4

    configs = [
        ("full_reflexion", "full"),
        ("no_self_tests", "no_tests"),
        ("no_reflection", "no_reflection"),
        ("no_memory", "no_memory"),
    ]

    all_results = {}
    for label, mode in configs:
        log("ABLATION", "RUNNING", f"Configuración: {label}")
        summary = run_ablation(label, mode, max_intentos=max_intentos, sample_size=sample_size)
        all_results[label] = summary

    report_path = os.path.join(RESULTS_DIR, "ablation_reflexion_report.json")
    with open(report_path, "w") as f:
        json.dump(all_results, f, indent=2)

    log("ABLATION", "DONE", f"Reporte guardado en {report_path}")
    print("\n" + "=" * 60)
    print(" RESUMEN DE ABLATION STUDIES — REFLEXION")
    print("=" * 60)
    for label, res in all_results.items():
        print(f" {label.ljust(25)} : pass@1={res['pass_at_1']:.2%}  pass@10={res['pass_at_10']:.2%}")
    print("=" * 60)
