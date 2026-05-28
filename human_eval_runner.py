import os
import sys
import json
import time
from datetime import datetime

from dotenv import load_dotenv
from human_eval.data import read_problems, write_jsonl
from human_eval.evaluation import evaluate_functional_correctness

load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reflexion import (
    log, ejecutar_agente_reflexion
)

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def extract_body_from_completion(codigo_actual, entry_point):
    """
    Extrae solo el cuerpo de la función (sin la firma/docstring).
    Devuelve el cuerpo con sangría de 4 espacios listo para HumanEval.
    """
    lines = codigo_actual.split("\n")
    body_lines = []
    found_def = False
    for line in lines:
        if f"def {entry_point}" in line:
            found_def = True
            continue
        if found_def:
            if line.strip() == "" and not body_lines:
                continue
            body_lines.append(line)
    if not body_lines:
        body_lines = lines[-3:]
    return "\n".join(body_lines)


def run_humaneval_problem(problem, max_intentos=4, use_self_generated_tests=True, output_jsonl=None):
    task_id = problem["task_id"]
    prompt_code = problem["prompt"]
    entry_point = problem["entry_point"]

    log("HUMANEVAL", "PROBLEM_START", f"{task_id} (entry_point={entry_point})")

    tarea = {
        "id": task_id,
        "descripcion": prompt_code,
        "tests": None,
    }

    exito, codigo_final = ejecutar_agente_reflexion(
        tarea,
        max_intentos=max_intentos,
        use_self_generated_tests=use_self_generated_tests,
        use_reflection=True,
        use_memory=True
    )

    body = extract_body_from_completion(codigo_final, entry_point)
    log("HUMANEVAL", "COMPLETION", f"{task_id} body ({len(body)} chars): {body[:200]}")

    if output_jsonl:
        output_jsonl.append({"task_id": task_id, "completion": body})

    return exito, body


def run_full_humaneval(max_intentos=4, sample_size=None, output_label="reflexion"):
    log("HUMANEVAL", "START", "Cargando HumanEval dataset")
    problems = read_problems()
    all_ids = sorted(problems.keys(), key=lambda x: int(x.split("/")[1]))

    if sample_size:
        all_ids = all_ids[:sample_size]
    log("HUMANEVAL", "START", f"Ejecutando {len(all_ids)} problemas ({max_intentos} intentos c/u)")

    output_path = os.path.join(RESULTS_DIR, f"{output_label}_samples.jsonl")
    results_path = os.path.join(RESULTS_DIR, f"{output_label}_results.json")
    jsonl_data = []

    total_start = time.time()
    solved = 0
    total_tokens = 0

    for idx, task_id in enumerate(all_ids):
        problem = problems[task_id]
        log("HUMANEVAL", "PROGRESS", f"[{idx+1}/{len(all_ids)}] {task_id}")
        try:
            exito, body = run_humaneval_problem(
                problem, max_intentos=max_intentos,
                use_self_generated_tests=True,
                output_jsonl=jsonl_data if output_label else None
            )
            if exito:
                solved += 1
        except Exception as e:
            log("HUMANEVAL", "ERROR", f"{task_id}: {str(e)}")
            jsonl_data.append({"task_id": task_id, "completion": ""})

    write_jsonl(output_path, jsonl_data)
    log("HUMANEVAL", "SAVED", f"Muestras guardadas en {output_path}")

    log("HUMANEVAL", "EVAL", "Evaluando con evaluate_functional_correctness...")
    eval_results = evaluate_functional_correctness(
        sample_file=output_path,
        k=[1, 10],
        n_workers=4,
        timeout=3.0
    )

    total_duration = time.time() - total_start
    pass_at_1 = eval_results.get("pass@1", 0)
    pass_at_10 = eval_results.get("pass@10", 0)

    summary = {
        "label": output_label,
        "total_problems": len(all_ids),
        "solved_in_loop": solved,
        "pass_at_1": pass_at_1,
        "pass_at_10": pass_at_10,
        "max_intentos": max_intentos,
        "total_duration_seconds": total_duration,
        "timestamp": datetime.now().isoformat(),
    }

    with open(results_path, "w") as f:
        json.dump(summary, f, indent=2)

    log("HUMANEVAL", "RESULTS", json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    label = sys.argv[1] if len(sys.argv) > 1 else "reflexion_full"
    sample = int(sys.argv[2]) if len(sys.argv) > 2 else None
    run_full_humaneval(max_intentos=4, sample_size=sample, output_label=label)
