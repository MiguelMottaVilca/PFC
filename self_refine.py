import os
from openai import OpenAI

# =====================================================================
# 0. CONFIGURACIÓN DEL CLIENTE
# =====================================================================
# Configura tu API key como variable de entorno:
# export OPENAI_API_KEY="tu-clave-aqui"
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def llamar_llm(prompt_text, model="gpt-3.5-turbo", temperature=0.7):
    """
    Función base para interactuar con la API de OpenAI.
    Se utiliza una temperatura de 0.7 tal como especifican los autores.
    """
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": prompt_text}
        ],
        temperature=temperature,
        max_tokens=300
    )
    return response.choices[0].message.content.strip()

def stop_condition(feedback, iteracion_actual, max_iter):
    """
    Función stop(fb_t, t) del Algoritmo 1.
    Se detiene si alcanza el límite de iteraciones o si el feedback es positivo.
    """
    if "already as negative as it can get" in feedback.lower() or "excellent work" in feedback.lower():
        return True
    if iteracion_actual >= max_iter - 1:
        return True
    return False

# =====================================================================
# 1. ALGORITMO SELF-REFINE (Fiel al Algoritmo 1 del paper)
# =====================================================================
def self_refine(x, max_iter=3):
    """
    Implementación matemática estricta de SELF-REFINE.
    x = Input (Entrada del usuario)
    """
    print(f"\n--- INICIANDO SELF-REFINE ---")
    print(f"Input x: {x}\n")

    # Prompts {p_gen, p_fb, p_refine}
    p_gen = "Rewrite the following review to have a Very Negative sentiment."
    p_fb = "Why is this review not Very negative? Identify specific positive words and suggest how to make it more negative."
    p_refine = "Okay, let's try again. Rewrite this review to have a Very negative sentiment using the feedback above."

    # Estructuras para almacenar los pasos en el tiempo 't'
    y = [None] * (max_iter + 1)  # Almacena y_0, y_1, ..., y_t
    fb = [None] * max_iter       # Almacena fb_0, fb_1, ..., fb_t-1

    # Línea 1: Inicialización -> y_0 = M(p_gen || x)
    prompt_gen = f"{p_gen}\n\nReview: {x}\nRewritten Review:"
    y[0] = llamar_llm(prompt_gen)
    print(f"[Iteración 0] y_0:\n{y[0]}\n")

    ultima_iteracion = 0

    # Línea 2: for iteration t en 0, 1, ... do
    for t in range(max_iter):
        ultima_iteracion = t
        print(f"--- Iteración t={t} ---")
        
        # Línea 3: Feedback -> fb_t = M(p_fb || x || y_t)
        # Nota: Ecuación 2 indica que el feedback se basa en 'x' y 'y_t' actual.
        prompt_fb = f"{p_fb}\n\nReview: {x}\nRewritten Review: {y[t]}\nFeedback:"
        fb[t] = llamar_llm(prompt_fb)
        print(f"fb_{t}:\n{fb[t]}\n")
        
        # Línea 4 y 5: if stop(fb_t, t) then break
        if stop_condition(fb[t], t, max_iter):
            print(">>> Condición de parada stop(·) activada.")
            break
            
        # Línea 7: Refine -> y_{t+1} = M(p_refine || x || y_0 || fb_0 || ... || y_t || fb_t)
        # Construimos la Ecuación 4 concatenando el historial exacto.
        historial_concatenado = f"Review: {x}\n"
        for i in range(t + 1):
            historial_concatenado += f"Rewritten Review: {y[i]}\n"
            historial_concatenado += f"Feedback: {fb[i]}\n"
            
        prompt_refine = f"{p_refine}\n\n{historial_concatenado}\nNew Rewritten Review:"
        y[t+1] = llamar_llm(prompt_refine)
        print(f"y_{t+1}:\n{y[t+1]}\n")
        
        # Actualizamos la última iteración válida si generamos una nueva
        ultima_iteracion = t + 1

    # Línea 10: return y_t (la última versión generada)
    return y[0], y[ultima_iteracion]

# =====================================================================
# 2. MÉTRICA DE EVALUACIÓN (GPT-4-pref proxy)
# =====================================================================
def gpt4_pref_evaluation(review_a, review_b):
    """
    Evaluación automatizada usando el modelo como juez ciego.
    """
    prompt_evaluacion = f"""Which review is more aligned with a 'Very Negative' sentiment?
Review A: {review_a}
Review B: {review_b}

Pick your answer from ['Review A', 'Review B', 'both', 'neither']. Generate a short explanation for your choice first.
Then, generate 'The more aligned review is A' or 'The more aligned review is B'."""

    # T=0 para evaluación determinista
    try:
        veredicto = llamar_llm(prompt_evaluacion, model="gpt-4", temperature=0.0)
    except Exception as e:
        print(f"(Fallback a GPT-3.5 para evaluar. Razón: {e})")
        veredicto = llamar_llm(prompt_evaluacion, model="gpt-3.5-turbo", temperature=0.0)
        
    return veredicto

# =====================================================================
# 3. EJECUCIÓN PRINCIPAL
# =====================================================================
if __name__ == "__main__":
    print("=" * 60)
    print(" SELF-REFINE ALGORITHM 1 (PAPER IMPLEMENTATION) ".center(60, "="))
    print("=" * 60)

    # Definimos x
    x = "The food was fantastic and the service was magical, an unforgettable experience!"
    
    # Ejecutamos Algoritmo 1
    y_0, y_final = self_refine(x, max_iter=3)
    
    print("\n" + "=" * 60)
    print(" RESULTADOS FINALES ".center(60, "="))
    print("=" * 60)
    print(f"[Baseline (y_0)]:\n{y_0}\n")
    print(f"[Self-Refine Final (y_t)]:\n{y_final}\n")

    print("\n" + "=" * 60)
    print(" EVALUACIÓN CIEGA (GPT-pref) ".center(60, "="))
    print("=" * 60)
    
    # Comparamos y_0 vs y_t
    evaluacion = gpt4_pref_evaluation(review_a=y_0, review_b=y_final)
    print(evaluacion)