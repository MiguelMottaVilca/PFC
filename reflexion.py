import os
import io
import json
import traceback
import contextlib
from datetime import datetime
from openai import OpenAI
import sys
from dotenv import load_dotenv

# ==========================================
# 0a. ANSI COLORS
# ==========================================
C_RESET = "\033[0m"
C_DIM = "\033[2m"
C_RED = "\033[31m"
C_GREEN = "\033[32m"
C_YELLOW = "\033[33m"
C_BLUE = "\033[34m"
C_MAGENTA = "\033[35m"
C_CYAN = "\033[36m"
C_BOLD_RED = "\033[1;31m"
C_BOLD_GREEN = "\033[1;32m"
C_BOLD_YELLOW = "\033[1;33m"
C_BOLD_BLUE = "\033[1;34m"
C_BOLD_MAGENTA = "\033[1;35m"
C_BOLD_CYAN = "\033[1;36m"
C_BOLD_WHITE = "\033[1;37m"

FASE_COLORS = {
    "MAIN": C_BOLD_WHITE,
    "ACTOR": C_BOLD_BLUE,
    "EVALUADOR": C_BOLD_GREEN,
    "REFLEXION": C_BOLD_YELLOW,
    "MEMORIA": C_BOLD_MAGENTA,
}

# ==========================================
# 0b. HELPER DE LOGGING
# ==========================================
def log(fase, tipo, mensaje):
    ts = datetime.now().strftime("%H:%M:%S")
    color = FASE_COLORS.get(fase, C_RESET)
    style = C_DIM if (tipo.endswith("_FULL") or tipo.endswith("_RAW")) else ""
    print(f"{style}{color}[{ts}][{fase}][{tipo}]{C_RESET} {style}{mensaje}{C_RESET}", flush=True)

def log_task_header(task_id, idx, total):
    w = 60
    print(f"\n{C_BOLD_WHITE}{'=' * w}{C_RESET}")
    print(f"{C_BOLD_WHITE}  TAREA {idx}/{total}: {task_id}{C_RESET}")
    print(f"{C_BOLD_WHITE}{'=' * w}{C_RESET}")

def log_attempt_header(attempt, max_attempts):
    w = 50
    print(f"\n{C_CYAN}{'─' * w}{C_RESET}")
    print(f"{C_CYAN}  Intento {attempt}/{max_attempts}{C_RESET}")
    print(f"{C_CYAN}{'─' * w}{C_RESET}")

class TeeStream:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for s in self.streams:
            s.write(data)
    def flush(self):
        for s in self.streams:
            s.flush()

# ==========================================
# 1. CONFIGURACIÓN Y PROMPTS
# ==========================================
# Carga variables de entorno desde .env
load_dotenv()

# Instancia el cliente de OpenAI. Requiere la variable de entorno OPENAI_API_KEY.
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
MODELO = "gpt-3.5-turbo" 

ACTOR_SYSTEM_PROMPT = """
You are a Python writing assistant. You will be given your previous implementation of a function, 
a series of unit tests results, and your self-reflection on your previous implementation. 
Apply the necessary changes below by responding only with the improved body of the function. 
Do not include the signature in your response. The first line of your response should have 
4 spaces of indentation so that it fits syntactically with the user provided signature.
"""

REFLEXION_SYSTEM_PROMPT = """
You are a Python writing assistant. You will be given your previous implementation of a function, 
a series of unit tests results, and your self-reflection on your previous implementation. 
Generate a short, concise verbal self-reflection on why the implementation failed the tests 
and exactly what needs to be changed in the code to fix it. Respond ONLY with the reflection.
"""

# ==========================================
# 2. CARGA DE TAREAS DESDE JSON
# ==========================================
def cargar_tareas():
    ruta = os.path.join(os.path.dirname(__file__), "tareas.json")
    with open(ruta, "r") as f:
        data = json.load(f)
    return [t for t in data["tareas"] if "descripcion" in t and "tests" in t]

# ==========================================
# 3. EVALUADOR (Me)
# ==========================================
def evaluar_codigo(codigo_generado, tests_unitarios):
    """
    Ejecuta el código generado y las pruebas unitarias en un entorno local.
    Retorna: (Booleano indicando éxito, Mensaje de error o éxito)
    """
    codigo_completo = f"{codigo_generado}\n\n{tests_unitarios}"
    log("EVALUADOR", "CODE_EXEC", f"Ejecutando {len(codigo_completo)} chars de código")
    log("EVALUADOR", "CODE_EXEC_FULL", codigo_completo)
    stdout_capture = io.StringIO()
    ambiente = {}

    try:
        with contextlib.redirect_stdout(stdout_capture):
            exec(codigo_completo, ambiente)
        stdout_text = stdout_capture.getvalue()
        if stdout_text:
            log("EVALUADOR", "STDOUT", stdout_text)
        log("EVALUADOR", "RESULT", "Success: All tests passed.")
        return True, "Success: All tests passed."
    
    except AssertionError:
        stdout_text = stdout_capture.getvalue()
        if stdout_text:
            log("EVALUADOR", "STDOUT", stdout_text)
        error_msg = traceback.format_exc()
        log("EVALUADOR", "RESULT", f"AssertionError: {error_msg[:300]}")
        return False, f"Test failed (AssertionError):\n{error_msg}"
    
    except Exception as e:
        stdout_text = stdout_capture.getvalue()
        if stdout_text:
            log("EVALUADOR", "STDOUT", stdout_text)
        error_msg = traceback.format_exc()
        log("EVALUADOR", "RESULT", f"Error: {str(e)[:200]}")
        return False, f"Execution/Syntax error:\n{error_msg}"

# ==========================================
# 4. AGENTE (Lógica de Reflexion)
# ==========================================
def llamar_modelo(system_prompt, user_prompt, fase_llamada="API"):
    """Interfaz estándar para comunicarse con la API de OpenAI."""
    prompt_len = len(system_prompt) + len(user_prompt)
    log(fase_llamada, "API_CALL", f"model={MODELO} temperature=0.5 prompt_len={prompt_len}")
    respuesta = client.chat.completions.create(
        model=MODELO,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.5 
    )
    contenido = respuesta.choices[0].message.content.strip()
    log(fase_llamada, "API_RESPONSE", f"len={len(contenido)} chars")
    log(fase_llamada, "API_RESPONSE_RAW", repr(contenido[:500]))
    return contenido

def ejecutar_agente_reflexion(tarea, max_intentos=4):
    """
    Implementación del bucle iterativo: Actor -> Evaluador -> Auto-reflexión.
    """
    descripcion = tarea["descripcion"]
    tests = tarea["tests"]
    
    log("MAIN", "TASK_START", f"Tarea: {tarea['id']}, max_intentos={max_intentos}")
    
    memoria_experiencias = [] # Memoria a largo plazo
    codigo_actual = ""
    
    for intento in range(max_intentos):
        log_attempt_header(intento + 1, max_intentos)
        log("MAIN", "ATTEMPT_START", f"Intento {intento + 1} de {max_intentos}")
        
        # --- FASE 1: ACTOR ---
        if intento == 0:
            prompt_actor = f"Task Signature:\n{descripcion}\n\nPlease implement the function body."
            log("ACTOR", "PROMPT", f"len={len(prompt_actor)} chars (primer intento, sin memoria)")
            log("ACTOR", "PROMPT_FULL", prompt_actor)
        else:
            memoria_texto = "\n".join(memoria_experiencias)
            prompt_actor = f"Task Signature:\n{descripcion}\n\nPrevious Code:\n{codigo_actual}\n\nSelf-Reflection (Lessons Learned):\n{memoria_texto}\n\nWrite the corrected function body."
            log("ACTOR", "PROMPT", f"len={len(prompt_actor)} chars (con memoria, {len(memoria_experiencias)} experiencias)")
            log("ACTOR", "PROMPT_FULL", prompt_actor)

        # El LLM genera el código
        respuesta_actor = llamar_modelo(ACTOR_SYSTEM_PROMPT, prompt_actor, fase_llamada="ACTOR")
        
        # Limpiar código (quitar fences) preservando indentación
        codigo_limpio = respuesta_actor.replace("```python", "").replace("```", "")
        # strip() completo borraría la indentación de la primera línea
        codigo_limpio = codigo_limpio.strip('\n')
        # Si la primera línea no tiene indentación, se la agregamos
        lineas = codigo_limpio.split('\n')
        if lineas and lineas[0] and not lineas[0].startswith(' '):
            lineas[0] = '    ' + lineas[0]
        codigo_limpio = '\n'.join(lineas)
        # Ensamblar código (Firma + Cuerpo generado)
        codigo_actual = f"{descripcion}\n{codigo_limpio}"
        
        log("ACTOR", "CODE_CLEAN", f"código limpio ({len(codigo_actual)} chars)")
        log("ACTOR", "CODE_CLEAN_FULL", codigo_actual)
        
        # --- FASE 2: EVALUADOR ---
        exito, feedback_error = evaluar_codigo(codigo_actual, tests)
        
        if exito:
            log("MAIN", "TASK_SUCCESS", f"Tarea {tarea['id']} completada en intento {intento + 1}")
            return True, codigo_actual, intento + 1
            
        log("MAIN", "ATTEMPT_FAIL", f"Intento {intento + 1} falló")
        
        # --- FASE 3: AUTO-REFLEXIÓN ---
        prompt_reflexion = f"Task Signature:\n{descripcion}\n\nFailed Code:\n{codigo_actual}\n\nUnit Test Error:\n{feedback_error}\n\nWhat went wrong?"
        log("REFLEXION", "PROMPT", f"len={len(prompt_reflexion)} chars")
        log("REFLEXION", "PROMPT_FULL", prompt_reflexion)
        nueva_reflexion = llamar_modelo(REFLEXION_SYSTEM_PROMPT, prompt_reflexion, fase_llamada="REFLEXION")
        
        log("REFLEXION", "RESULT", nueva_reflexion)
        
        # --- FASE 4: ACTUALIZAR MEMORIA ---
        memoria_experiencias.append(f"[Trial {intento+1}] {nueva_reflexion}")
        log("MEMORIA", "UPDATE", f"Experiencias totales: {len(memoria_experiencias)}")
        log("MEMORIA", "STATE", f"{memoria_experiencias}")
        
        # Limitar la memoria a las últimas 3 experiencias para mantener el contexto limpio
        if len(memoria_experiencias) > 3:
            memoria_experiencias.pop(0)
            log("MEMORIA", "TRIM", "Memoria truncada a últimas 3 experiencias")

    log("MAIN", "TASK_FAIL", f"Tarea {tarea['id']} fallida tras {max_intentos} intentos.")
    return False, codigo_actual, max_intentos

# ==========================================
# 5. PUNTO DE ENTRADA PRINCIPAL
# ==========================================
if __name__ == "__main__":
    # Verificación de seguridad para la API Key
    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR CRÍTICO: No se encontró la variable de entorno 'OPENAI_API_KEY'.")
        print("Antes de ejecutar el script, por favor configúrala.")
        print("Ejemplo en Linux/Mac: export OPENAI_API_KEY='tu_clave_aqui'")
        print("Ejemplo en Windows:   set OPENAI_API_KEY='tu_clave_aqui'")
        exit(1)

    os.makedirs("logs", exist_ok=True)
    log_path = f"logs/reflexion_{datetime.now():%Y%m%d_%H%M%S}.log"
    log_file = open(log_path, "w")
    sys.stdout = TeeStream(sys.__stdout__, log_file)
    print(f"{C_BOLD_CYAN}Log guardado en: {log_path}{C_RESET}")

    resultados = []
    
    # Procesar cada tarea del dataset
    tareas = cargar_tareas()
    for idx, tarea in enumerate(tareas, 1):
        log_task_header(tarea["id"], idx, len(tareas))
        exito, codigo_final, iteracion = ejecutar_agente_reflexion(tarea)
        resultados.append({"tarea": tarea["id"], "exito": exito, "iteracion": iteracion})
        
    # Mostrar resumen final
    print(f"\n\n{C_BOLD_WHITE}{'='*45}{C_RESET}")
    print(f"{C_BOLD_WHITE} RESUMEN DE EVALUACIÓN FINAL {C_RESET}")
    print(f"{C_BOLD_WHITE}{'='*45}{C_RESET}")
    for res in resultados:
        if res["exito"]:
            estado = f"{C_BOLD_GREEN}✅ PASÓ (iter {res['iteracion']}){C_RESET}"
        else:
            estado = f"{C_BOLD_RED}❌ FALLÓ ({res['iteracion']} intentos){C_RESET}"
        print(f" {C_BOLD_WHITE}{res['tarea'].ljust(25)}{C_RESET} : {estado}")
    print(f"{C_BOLD_WHITE}{'='*45}{C_RESET}")