import os
import io
import traceback
import contextlib
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv

# ==========================================
# 0. HELPER DE LOGGING
# ==========================================
def log(fase, tipo, mensaje):
    ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"[{ts}] [{fase}] [{tipo}] {mensaje}")

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
# 2. ENTORNO / TAREAS (DATASET)
# ==========================================
DATASET_TAREAS = [
    {
        "id": "HumanEval_Ejemplo1",
        "descripcion": """
def minSubArraySum(nums):
    # Given an array of integers nums, find the minimum sum of any non-empty sub-array of nums.
    # Example: minSubArraySum([2, 3, 4, 1, 2, 4]) == 1
    # Example: minSubArraySum([-1, -2, -3]) == -6
""",
        "tests": """
assert minSubArraySum([2, 3, 4, 1, 2, 4]) == 1
assert minSubArraySum([-1, -2, -3]) == -6
assert minSubArraySum([5, 5, 5]) == 5
assert minSubArraySum([3, -4, 2, -3]) == -5
"""
    },
    {
        "id": "Strings_Parentesis",
        "descripcion": """
def match_parens(lst):
    # You are given a list of two strings of open '(' or close ')' parentheses only.
    # Return 'Yes' if there is a way to concatenate them to make a valid sequence, else 'No'.
    # Example: match_parens(['()(', ')']) == 'Yes'
""",
        "tests": """
assert match_parens(['()(', ')']) == 'Yes'
assert match_parens([')', ')']) == 'No'
assert match_parens(['((((', '((())']) == 'No'
assert match_parens(['()', '()']) == 'Yes'
"""
    }
]

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
        
        # Ensamblar código (Firma + Cuerpo generado)
        codigo_limpio = respuesta_actor.replace("```python", "").replace("```", "").strip()
        codigo_actual = f"{descripcion}\n{codigo_limpio}"
        
        log("ACTOR", "CODE_CLEAN", f"código limpio ({len(codigo_actual)} chars)")
        log("ACTOR", "CODE_CLEAN_FULL", codigo_actual)
        
        # --- FASE 2: EVALUADOR ---
        exito, feedback_error = evaluar_codigo(codigo_actual, tests)
        
        if exito:
            log("MAIN", "TASK_SUCCESS", f"Tarea {tarea['id']} completada en intento {intento + 1}")
            return True, codigo_actual
            
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
    return False, codigo_actual

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

    resultados = []
    
    # Procesar cada tarea del dataset
    for tarea in DATASET_TAREAS:
        exito, codigo_final = ejecutar_agente_reflexion(tarea)
        resultados.append({"tarea": tarea["id"], "exito": exito})
        
    # Mostrar resumen final
    print("\n\n" + "="*40)
    print(" RESUMEN DE EVALUACIÓN FINAL ")
    print("="*40)
    for res in resultados:
        estado = "✅ PASÓ" if res["exito"] else "❌ FALLÓ"
        print(f" {res['tarea'].ljust(25)} : {estado}")
    print("="*40)