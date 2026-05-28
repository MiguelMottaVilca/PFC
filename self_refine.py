import os
import json
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv

# =====================================================================
# Default format templates for prompt construction
# =====================================================================

CODE_FORMAT = {
    "input_format": "Problem:\n{x}\nImplementation:",
    "fb_format": "Problem:\n{x}\nImplementation:\n{y}\nFeedback:",
    "history_format": "Problem: {x}\n",
    "history_entry": "Implementation: {y}\nFeedback: {fb}\n",
    "refine_format": "New Implementation:",
}

TEXT_FORMAT = {
    "input_format": "Review: {x}\nRewritten Review:",
    "fb_format": "Review: {x}\nRewritten Review: {y}\nFeedback:",
    "history_format": "Review: {x}\n",
    "history_entry": "Rewritten Review: {y}\nFeedback: {fb}\n",
    "refine_format": "New Rewritten Review:",
}

# =====================================================================
# 0. HELPER DE LOGGING
# =====================================================================

def log(fase, tipo, mensaje):
    ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"[{ts}] [{fase}] [{tipo}] {mensaje}")

# =====================================================================
# 0. CONFIGURACIÓN DEL CLIENTE
# =====================================================================

load_dotenv()
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def llamar_llm(prompt_text, model="gpt-3.5-turbo", temperature=0.7, fase_llamada="API"):
    log(fase_llamada, "API_CALL", f"model={model} temperature={temperature} max_tokens=300 prompt_len={len(prompt_text)}")
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": prompt_text}
        ],
        temperature=temperature,
        max_tokens=300
    )
    contenido = response.choices[0].message.content.strip()
    log(fase_llamada, "API_RESPONSE", f"len={len(contenido)} chars")
    log(fase_llamada, "API_RESPONSE_RAW", repr(contenido[:500]))
    return contenido

def stop_condition(feedback, iteracion_actual, max_iter):
    stop_por_keyword = "already as negative as it can get" in feedback.lower() or "excellent work" in feedback.lower()
    stop_por_limite = iteracion_actual >= max_iter - 1
    log("STOP", "CHECK", f"keywords={stop_por_keyword} limite_iter={stop_por_limite} iter={iteracion_actual}/{max_iter-1}")
    if stop_por_keyword:
        log("STOP", "TRIGGERED", "Feedback contiene keyword de parada")
        return True
    if stop_por_limite:
        log("STOP", "TRIGGERED", f"Alcanzado límite de iteraciones ({max_iter})")
        return True
    return False

# =====================================================================
# 1. CARGA DE TAREAS DESDE JSON
# =====================================================================

def cargar_tareas():
    ruta = os.path.join(os.path.dirname(__file__), "tareas.json")
    with open(ruta, "r") as f:
        data = json.load(f)
    return [t for t in data["tareas"] if "input" in t]

# =====================================================================
# 2. ALGORITMO SELF-REFINE (Fiel al Algoritmo 1 del paper)
# =====================================================================

def self_refine(x, task_config, max_iter=3):
    p_gen = task_config.get("p_gen", "Rewrite the following review to have a Very Negative sentiment.")
    p_fb = task_config.get("p_fb", "Why is this review not Very negative? Identify specific positive words and suggest how to make it more negative.")
    p_refine = task_config.get("p_refine", "Okay, let's try again. Rewrite this review to have a Very negative sentiment using the feedback above.")

    is_code = "tests" in task_config
    fmt = CODE_FORMAT if is_code else TEXT_FORMAT

    inp_fmt = task_config.get("input_format", fmt["input_format"])
    fb_fmt = task_config.get("fb_format", fmt["fb_format"])
    hist_fmt = task_config.get("history_format", fmt["history_format"])
    hist_entry = task_config.get("history_entry", fmt["history_entry"])
    ref_fmt = task_config.get("refine_format", fmt["refine_format"])

    log("MAIN", "SELF_REFINE_START", f"Input x: {x[:100]}..., max_iter={max_iter}")

    y = [None] * (max_iter + 1)
    fb = [None] * max_iter

    prompt_gen = f"{p_gen}\n\n{inp_fmt.replace('{x}', x)}"
    log("GENERATE", "PROMPT", f"len={len(prompt_gen)} chars")
    log("GENERATE", "PROMPT_FULL", prompt_gen)
    y[0] = llamar_llm(prompt_gen, fase_llamada="GENERATE")
    log("GENERATE", "RESULT", f"y_0 ({len(y[0])} chars): {y[0][:200]}")

    ultima_iteracion = 0

    for t in range(max_iter):
        ultima_iteracion = t
        log("MAIN", "ITERATION_START", f"t={t}")

        prompt_fb = f"{p_fb}\n\n{fb_fmt.replace('{x}', x).replace('{y}', y[t])}"
        log("FEEDBACK", "PROMPT", f"t={t} len={len(prompt_fb)} chars")
        log("FEEDBACK", "PROMPT_FULL", prompt_fb)
        fb[t] = llamar_llm(prompt_fb, fase_llamada="FEEDBACK")
        log("FEEDBACK", "RESULT", f"fb_{t} ({len(fb[t])} chars): {fb[t][:200]}")

        if stop_condition(fb[t], t, max_iter):
            log("MAIN", "STOP_BREAK", f"Bucle detenido en t={t}")
            break

        historial = hist_fmt.replace("{x}", x)
        for i in range(t + 1):
            historial += hist_entry.replace("{y}", y[i]).replace("{fb}", fb[i])

        prompt_refine = f"{p_refine}\n\n{historial}\n{ref_fmt}"
        log("REFINE", "PROMPT", f"t={t} len={len(prompt_refine)} chars")
        log("REFINE", "PROMPT_FULL", prompt_refine)
        y[t+1] = llamar_llm(prompt_refine, fase_llamada="REFINE")
        log("REFINE", "RESULT", f"y_{t+1} ({len(y[t+1])} chars): {y[t+1][:200]}")

        ultima_iteracion = t + 1

    log("MAIN", "SELF_REFINE_END", f"y_0 ({len(y[0])} chars) y y_{ultima_iteracion} ({len(y[ultima_iteracion])} chars)")
    return y[0], y[ultima_iteracion]

# =====================================================================
# 2. MÉTRICA DE EVALUACIÓN (GPT-4-pref proxy)
# =====================================================================

def gpt4_pref_evaluation(output_a, output_b, task_name="this task"):
    prompt_evaluacion = f"""Which output is better for {task_name}?
Output A: {output_a}
Output B: {output_b}

Pick your answer from ['Output A', 'Output B', 'both', 'neither']. Generate a short explanation for your choice first.
Then, generate 'The better output is A' or 'The better output is B'."""

    log("EVAL", "PROMPT", f"len={len(prompt_evaluacion)} chars")
    log("EVAL", "PROMPT_FULL", prompt_evaluacion)

    try:
        veredicto = llamar_llm(prompt_evaluacion, model="gpt-4", temperature=0.0, fase_llamada="EVAL")
    except Exception as e:
        log("EVAL", "FALLBACK", f"GPT-4 falló: {e}. Usando GPT-3.5")
        veredicto = llamar_llm(prompt_evaluacion, model="gpt-3.5-turbo", temperature=0.0, fase_llamada="EVAL")

    log("EVAL", "VEREDICT", veredicto)
    return veredicto

# =====================================================================
# 2b. EXTRACCIÓN DE CUERPO DE FUNCIÓN DESDE SALIDA DEL LLM
# =====================================================================

def extraer_cuerpo_funcion(output):
    texto = output.replace("```python", "").replace("```", "")
    lineas = texto.split('\n')
    idx_def = None
    for i, linea in enumerate(lineas):
        if linea.strip().startswith('def ') and linea.strip().endswith(':'):
            idx_def = i
            break
    if idx_def is not None:
        cuerpo = []
        for linea in lineas[idx_def+1:]:
            if linea.strip() == '' or linea.startswith(' ') or linea.startswith('\t') or linea.strip().startswith('#'):
                cuerpo.append(linea)
            elif cuerpo and not linea.strip():
                cuerpo.append(linea)
            elif not cuerpo and not linea.strip():
                continue
            else:
                break
        if not cuerpo:
            cuerpo = [linea for linea in lineas[idx_def+1:] if linea.strip()]
        return '\n'.join(cuerpo)
    lineas_util = [l for l in lineas if l.strip()]
    return '\n'.join(lineas_util) if lineas_util else texto.strip()

# =====================================================================
# 3. EVALUADOR LOCAL (similar a reflexion.evaluar_codigo)
# =====================================================================

def evaluar_codigo_local(codigo, tests):
    import io
    import traceback
    import contextlib
    codigo_completo = f"{codigo}\n\n{tests}"
    stdout_capture = io.StringIO()
    ambiente = {}

    try:
        with contextlib.redirect_stdout(stdout_capture):
            exec(codigo_completo, ambiente)
        return True, "Success: All tests passed."
    except AssertionError:
        error_msg = traceback.format_exc()
        return False, f"Test failed (AssertionError):\n{error_msg}"
    except Exception as e:
        error_msg = traceback.format_exc()
        return False, f"Execution/Syntax error:\n{error_msg}"

# =====================================================================
# 3. EJECUCIÓN PRINCIPAL
# =====================================================================

if __name__ == "__main__":
    print("=" * 60)
    print(" SELF-REFINE ALGORITHM 1 (PAPER IMPLEMENTATION) ".center(60, "="))
    print("=" * 60)

    tareas = cargar_tareas()
    resultados = []

    for tarea in tareas:
        print("\n" + "=" * 60)
        print(f" TAREA: {tarea['id']} ".center(60, "="))
        print("=" * 60)

        x = tarea["input"]

        y_0, y_final = self_refine(x, tarea, max_iter=tarea.get("max_iter", 3))

        print("\n" + "=" * 60)
        print(" RESULTADOS FINALES ".center(60, "="))
        print("=" * 60)
        print(f"[Baseline (y_0)]:\n{y_0}\n")
        print(f"[Self-Refine Final (y_t)]:\n{y_final}\n")

        if "tests" in tarea and "descripcion" in tarea:
            cuerpo = extraer_cuerpo_funcion(y_final)
            if cuerpo and not cuerpo.startswith(" "):
                cuerpo = "\n".join(
                    "    " + line if line.strip() else line
                    for line in cuerpo.split("\n")
                )
            codigo_completo = f"{tarea['descripcion']}\n{cuerpo}"
            exito, msg = evaluar_codigo_local(codigo_completo, tarea["tests"])
            estado = "PASÓ" if exito else "FALLÓ"
            print(f" TESTS: {estado}")
            resultados.append({"tarea": tarea["id"], "exito": exito})

        print("\n" + "=" * 60)
        print(" EVALUACIÓN CIEGA (GPT-pref) ".center(60, "="))
        print("=" * 60)

        evaluacion = gpt4_pref_evaluation(y_0, y_final, task_name=tarea["id"])
        print(evaluacion)

    if resultados:
        print("\n\n" + "=" * 40)
        print(" RESUMEN DE EVALUACIÓN (TESTS) ")
        print("=" * 40)
        for res in resultados:
            estado = "✅ PASÓ" if res["exito"] else "❌ FALLÓ"
            print(f" {res['tarea'].ljust(25)} : {estado}")
        print("=" * 40)
