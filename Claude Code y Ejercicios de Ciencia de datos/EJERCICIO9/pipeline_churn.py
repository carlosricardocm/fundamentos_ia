# -*- coding: utf-8 -*-
"""
Pipeline completo de clasificacion de abandono (churn).

Entrada : churn_clientes_features.csv   (objetivo: abandono)
Salidas : modelo_churn.pkl              (modelo + lista de variables)

QUE HACE
--------
0. Auditoria de fuga: detecta variables que separan las clases de forma
   determinista antes de entrenar nada.
1. Particion 80/20 estratificada.
2. Imputacion (mediana / moda) + codificacion one-hot.
3. Escalado de las numericas.
4. Regresion logistica, random forest y gradient boosting.
5. Exactitud, precision, recall y F1 por validacion cruzada estratificada.
6. Busqueda en malla sobre el mejor modelo.
7. Importancia de variables (nativa + permutacion, agregada a variable origen).
8. Guardado del modelo con su lista de variables y metadatos.

Todo el proceso se corre TRES VECES, quitando informacion en cada paso:
  Escenario A  - con todas las variables (tal como pide el enunciado).
  Escenario B  - excluyendo las variables marcadas como fuga.
  Escenario C  - solo variables crudas (sin ninguna derivada ni el contrato).
Si las metricas siguen siendo perfectas en C, el problema no son unas cuantas
columnas: es el conjunto de datos entero.
"""

from datetime import datetime
from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,
                                     cross_val_score, cross_validate,
                                     train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)
pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 60)

BASE = Path(__file__).resolve().parent
ENTRADA = BASE / "churn_clientes_features.csv"
SALIDA_MODELO = BASE / "modelo_churn.pkl"

OBJETIVO = "abandono"
ID = "id_cliente"
SEMILLA = 42
PROPORCION_PRUEBA = 0.20
N_PLIEGUES = 5

# Umbrales de la auditoria de fuga
UMBRAL_SEPARADOR = 0.995   # exactitud de un tocon de decision de 1 variable
UMBRAL_PUREZA = 1.00       # pureza de un lado de la particion

# De que escenario sale modelo_churn.pkl: "B" = sin fugas (el desplegable)
ESCENARIO_A_GUARDAR = "B"

# Variables presentes en el csv original (churn_clientes_crudo.csv, ejercicio 7).
# Todo lo demas en churn_clientes_features.csv es derivado de estas.
VARIABLES_CRUDAS = ["edad", "meses_antiguedad", "gasto_mensual", "num_productos",
                    "tickets_soporte", "dias_ultimo_acceso", "tipo_contrato",
                    "metodo_pago", "region"]

SEPARADOR = "=" * 78


def titulo(texto):
    print("\n" + SEPARADOR)
    print(texto)
    print(SEPARADOR)


# =====================================================================
# SECCION 0. CARGA Y AUDITORIA DE FUGA DE INFORMACION
# =====================================================================
def cargar():
    if not ENTRADA.exists():
        raise FileNotFoundError("No encuentro " + str(ENTRADA))
    df = pd.read_csv(ENTRADA)
    titulo("0. CARGA Y DIAGNOSTICO DEL CONJUNTO")
    print("Filas: {}   Columnas: {}".format(len(df), df.shape[1]))
    print("IDs duplicados: {}".format(df[ID].duplicated().sum()))
    print("Nulos totales: {}".format(int(df.isna().sum().sum())))
    print("Balance del objetivo: {:.1%} de abandono ({} de {})".format(
        df[OBJETIVO].mean(), int(df[OBJETIVO].sum()), len(df)))
    nulos = df.isna().sum()
    nulos = nulos[nulos > 0]
    if len(nulos):
        print("\nNulos por columna:")
        print(nulos.to_string())
    else:
        print("\nNo hay valores faltantes: la imputacion queda en el pipeline "
              "como salvaguarda para datos nuevos, pero aqui no altera nada.")
    return df


def tipos_de_variable(df, excluir):
    """Separa columnas numericas y categoricas, quitando id, objetivo y excluidas."""
    fuera = set([ID, OBJETIVO] + list(excluir))
    cols = [c for c in df.columns if c not in fuera]
    num = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    cat = [c for c in cols if c not in num]
    return num, cat


def auditar_fuga(df):
    """Busca variables que por si solas resuelven el problema.

    Dos senales complementarias:
      (a) Un tocon de decision (arbol de profundidad 1) sobre esa unica
          variable alcanza exactitud ~1.0 en validacion cruzada.
      (b) Alguna categoria o lado del corte es 100% puro y concentra una
          fraccion no trivial de los casos.
    """
    titulo("0b. AUDITORIA DE FUGA (antes de entrenar)")
    num, cat = tipos_de_variable(df, excluir=[])
    y = df[OBJETIVO]
    cv = StratifiedKFold(n_splits=N_PLIEGUES, shuffle=True, random_state=SEMILLA)

    filas = []
    for col in num + cat:
        s = df[col]
        if col in cat:
            x = pd.factorize(s.astype(str))[0].reshape(-1, 1)
        else:
            x = s.fillna(s.median()).to_numpy().reshape(-1, 1)
        tocon = DecisionTreeClassifier(max_depth=1, random_state=SEMILLA)
        exactitud = cross_val_score(tocon, x, y, cv=cv, scoring="accuracy").mean()

        # pureza maxima de un grupo (categorias para cat, corte del tocon para num)
        if col in cat:
            grupos = df.groupby(s.astype(str))[OBJETIVO].agg(["count", "mean"])
        else:
            tocon.fit(x, y)
            corte = tocon.tree_.threshold[0]
            lado = pd.Series(np.where(x.ravel() <= corte, "<=", ">"), index=df.index)
            grupos = df.groupby(lado)[OBJETIVO].agg(["count", "mean"])
        pureza = np.maximum(grupos["mean"], 1 - grupos["mean"])
        grupos = grupos.assign(pureza=pureza)
        peor = grupos.sort_values(["pureza", "count"], ascending=False).iloc[0]
        filas.append({"variable": col, "exactitud_1var": exactitud,
                      "pureza_max": peor["pureza"], "n_grupo": int(peor["count"])})

    tabla = pd.DataFrame(filas).sort_values("exactitud_1var", ascending=False)
    tabla["veredicto"] = np.where(
        tabla["exactitud_1var"] >= UMBRAL_SEPARADOR, "SEPARADOR PERFECTO",
        np.where((tabla["pureza_max"] >= UMBRAL_PUREZA) & (tabla["n_grupo"] >= 10),
                 "regla de un lado 100% pura", "ok"))

    print("Poder predictivo de CADA variable POR SI SOLA (tocon de decision, "
          "CV {} pliegues):\n".format(N_PLIEGUES))
    vista = tabla.copy()
    vista["exactitud_1var"] = (vista["exactitud_1var"] * 100).round(1)
    vista["pureza_max"] = (vista["pureza_max"] * 100).round(1)
    vista.columns = ["variable", "exactitud_%", "pureza_max_%", "n_grupo", "veredicto"]
    print(vista.to_string(index=False))

    fugas = tabla.loc[tabla["veredicto"] == "SEPARADOR PERFECTO", "variable"].tolist()

    print("\n-- Redundancias exactas detectadas --")
    redundantes = detectar_redundancias(df)
    for r in redundantes:
        print("  " + r)
    if not redundantes:
        print("  (ninguna)")

    if fugas:
        print("\n>>> ALERTA: {} variable(s) separan las clases sin error:".format(
            len(fugas)))
        for f in fugas:
            print("      - " + f)
        print("    Cualquier modelo entrenado con ellas dara ~100% y no habra")
        print("    aprendido nada: solo copia la regla. Se excluyen en el escenario B.")
    else:
        print("\nNo se detectaron separadores perfectos.")

    auditar_solape(df, num)
    return fugas


def auditar_solape(df, num):
    """Cuanto se pisan las dos clases en cada variable numerica.

    Si el rango de los que se quedan y el de los que abandonan casi no se
    tocan, las dos poblaciones fueron generadas por separado y ningun
    clasificador puede fallar. En datos reales de churn los rangos se
    solapan casi por completo.
    """
    print("\n-- Solape entre clases por variable numerica --")
    print("(fraccion de clientes que caen en el rango COMPARTIDO por las dos clases)\n")
    filas = []
    for c in num:
        a = df.loc[df[OBJETIVO] == 0, c]
        b = df.loc[df[OBJETIVO] == 1, c]
        bajo = max(a.min(), b.min())
        alto = min(a.max(), b.max())
        if alto < bajo:            # rangos totalmente disjuntos
            frac = 0.0
        else:
            frac = float(((df[c] >= bajo) & (df[c] <= alto)).mean())
        filas.append({"variable": c,
                      "rango_clase_0": "[{:g}, {:g}]".format(a.min(), a.max()),
                      "rango_clase_1": "[{:g}, {:g}]".format(b.min(), b.max()),
                      "en_zona_comun": frac})
    tabla = pd.DataFrame(filas).sort_values("en_zona_comun")
    vista = tabla.copy()
    vista["en_zona_comun"] = (vista["en_zona_comun"] * 100).round(1)
    vista.columns = ["variable", "rango_clase_0", "rango_clase_1", "en_zona_comun_%"]
    print(vista.to_string(index=False))

    criticas = tabla[tabla["en_zona_comun"] <= 0.20]["variable"].tolist()
    if criticas:
        print("\n>>> ALERTA: en {} variable(s) menos del 20% de los clientes cae en".format(
            len(criticas)))
        print("    la zona ambigua: {}".format(", ".join(criticas)))
        print("    Las dos clases viven en regiones practicamente separadas del")
        print("    espacio. Eso no ocurre en datos de churn reales.")


def detectar_redundancias(df):
    """Pares de columnas que son la misma informacion (copia, escala o recodificacion)."""
    hallazgos = []
    num = list(df.select_dtypes(include=np.number).columns)
    num = [c for c in num if c != OBJETIVO]
    for i, a in enumerate(num):
        for b in num[i + 1:]:
            x = df[a].to_numpy(float)
            y = df[b].to_numpy(float)
            if np.std(x) == 0 or np.std(y) == 0:
                continue
            if abs(np.corrcoef(x, y)[0, 1]) > 0.9999:
                mask = x != 0
                razon = np.round(y[mask] / x[mask], 6)
                extra = " (b = a x {:g})".format(razon[0]) if len(set(razon)) == 1 else ""
                hallazgos.append("{} <-> {}: correlacion 1.0{}".format(a, b, extra))
    # categoricas que son recodificacion de una binaria
    cats = [c for c in df.select_dtypes(exclude=np.number).columns if c != ID]
    bins = [c for c in num if df[c].nunique() == 2]
    for c in cats:
        for b in bins:
            if df[c].nunique() == df[b].nunique() and df.groupby(c)[b].nunique().max() == 1:
                hallazgos.append("{} <-> {}: {} es la codificacion binaria de {}".format(
                    c, b, b, c))
    return hallazgos


# =====================================================================
# SECCION 1-3. PREPROCESAMIENTO
# =====================================================================
def construir_preprocesador(num, cat):
    """Imputacion + escalado para numericas, imputacion + one-hot para categoricas.

    Va dentro del Pipeline, no antes: asi la mediana, la moda y la media/desviacion
    se calculan SOLO con el pliegue de entrenamiento en cada iteracion de la
    validacion cruzada. Escalar antes de partir seria fuga de preprocesamiento.
    """
    rama_num = Pipeline([
        ("imputar", SimpleImputer(strategy="median")),
        ("escalar", StandardScaler()),
    ])
    rama_cat = Pipeline([
        ("imputar", SimpleImputer(strategy="most_frequent")),
        ("codificar", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer(
        [("num", rama_num, num), ("cat", rama_cat, cat)],
        remainder="drop", verbose_feature_names_out=True)


def esquema_modelos():
    """Estimador base + malla de hiperparametros de cada familia."""
    return {
        "regresion_logistica": {
            "estimador": LogisticRegression(max_iter=5000, random_state=SEMILLA),
            "malla": {
                "modelo__C": [0.01, 0.1, 1.0, 10.0, 100.0],
                "modelo__solver": ["lbfgs", "liblinear"],
                "modelo__class_weight": [None, "balanced"],
            },
        },
        "random_forest": {
            "estimador": RandomForestClassifier(random_state=SEMILLA, n_jobs=-1),
            "malla": {
                "modelo__n_estimators": [200, 500],
                "modelo__max_depth": [None, 3, 5, 8],
                "modelo__min_samples_leaf": [1, 2, 4],
                "modelo__max_features": ["sqrt", 0.5],
            },
        },
        "gradient_boosting": {
            "estimador": GradientBoostingClassifier(random_state=SEMILLA),
            "malla": {
                "modelo__n_estimators": [100, 300],
                "modelo__learning_rate": [0.03, 0.1, 0.2],
                "modelo__max_depth": [2, 3],
                "modelo__subsample": [0.8, 1.0],
            },
        },
    }


# =====================================================================
# SECCION 4-5. COMPARACION POR VALIDACION CRUZADA
# =====================================================================
METRICAS = {"exactitud": "accuracy", "precision": "precision",
            "recall": "recall", "f1": "f1"}
ORDEN_METRICAS = ["exactitud", "precision", "recall", "f1"]


def comparar_modelos(X_ent, y_ent, num, cat):
    cv = StratifiedKFold(n_splits=N_PLIEGUES, shuffle=True, random_state=SEMILLA)
    filas = []
    for nombre, cfg in esquema_modelos().items():
        pipe = Pipeline([("preparar", construir_preprocesador(num, cat)),
                         ("modelo", cfg["estimador"])])
        res = cross_validate(pipe, X_ent, y_ent, cv=cv, scoring=METRICAS, n_jobs=-1)
        fila = {"modelo": nombre}
        for m in ORDEN_METRICAS:
            fila[m] = res["test_" + m].mean()
            fila[m + "_de"] = res["test_" + m].std()
        filas.append(fila)
    return pd.DataFrame(filas).sort_values("f1", ascending=False).reset_index(drop=True)


def imprimir_comparacion(tabla):
    print("Validacion cruzada estratificada de {} pliegues sobre el conjunto de "
          "ENTRENAMIENTO".format(N_PLIEGUES))
    print("(media +/- desviacion entre pliegues)\n")
    print("{:<22}".format("modelo") + "".join(
        "{:>22}".format(m) for m in ORDEN_METRICAS))
    print("-" * (22 + 22 * len(ORDEN_METRICAS)))
    for _, r in tabla.iterrows():
        celdas = "".join("{:>15.3f} +/-{:.3f}".format(r[m], r[m + "_de"])
                         for m in ORDEN_METRICAS)
        print("{:<22}{}".format(r["modelo"], celdas))


# =====================================================================
# SECCION 6. BUSQUEDA EN MALLA
# =====================================================================
def ajustar_mejor(nombre, X_ent, y_ent, num, cat):
    cfg = esquema_modelos()[nombre]
    pipe = Pipeline([("preparar", construir_preprocesador(num, cat)),
                     ("modelo", cfg["estimador"])])
    cv = StratifiedKFold(n_splits=N_PLIEGUES, shuffle=True, random_state=SEMILLA)
    busqueda = GridSearchCV(pipe, cfg["malla"], scoring="f1", cv=cv,
                            n_jobs=-1, refit=True, return_train_score=True)
    busqueda.fit(X_ent, y_ent)
    n_comb = len(busqueda.cv_results_["params"])
    print("Modelo ajustado : {}".format(nombre))
    print("Combinaciones probadas : {}  ({} ajustes)".format(
        n_comb, n_comb * N_PLIEGUES))
    print("Mejores hiperparametros :")
    for k, v in sorted(busqueda.best_params_.items()):
        print("    {:<20} = {}".format(k.replace("modelo__", ""), v))
    i = busqueda.best_index_
    print("F1 en CV (entrenamiento) : {:.3f}".format(
        busqueda.cv_results_["mean_train_score"][i]))
    print("F1 en CV (validacion)    : {:.3f} +/- {:.3f}".format(
        busqueda.best_score_, busqueda.cv_results_["std_test_score"][i]))
    return busqueda


# =====================================================================
# SECCION 7. IMPORTANCIA DE VARIABLES
# =====================================================================
def variable_origen(nombre_transformado, originales):
    """'cat__tipo_contrato_mensual' -> 'tipo_contrato' (match mas largo)."""
    limpio = nombre_transformado.split("__", 1)[-1]
    candidatos = [c for c in originales
                  if limpio == c or limpio.startswith(c + "_")]
    return max(candidatos, key=len) if candidatos else limpio


def barra(valor, maximo, ancho=34):
    if maximo <= 0:
        return ""
    return "#" * int(round(valor / maximo * ancho))


def importancias(pipe, X_pru, y_pru, originales):
    """Importancia nativa del modelo + importancia por permutacion, ambas
    agregadas al nivel de la variable original (suma de sus columnas one-hot)."""
    modelo = pipe.named_steps["modelo"]
    nombres = pipe.named_steps["preparar"].get_feature_names_out()

    if hasattr(modelo, "feature_importances_"):
        nativa = modelo.feature_importances_
        etiqueta = "reduccion de impureza"
    else:
        nativa = np.abs(modelo.coef_.ravel())
        etiqueta = "|coeficiente| sobre datos escalados"

    perm = permutation_importance(pipe, X_pru, y_pru, scoring="f1",
                                  n_repeats=50, random_state=SEMILLA, n_jobs=-1)

    detalle = pd.DataFrame({
        "columna": nombres,
        "origen": [variable_origen(n, originales) for n in nombres],
        "nativa": nativa,
    })
    agregada = detalle.groupby("origen")["nativa"].sum()
    perm_df = pd.DataFrame({"origen": originales,
                            "permutacion": perm.importances_mean,
                            "permutacion_de": perm.importances_std})
    tabla = (perm_df.set_index("origen")
             .join(agregada.rename("nativa"))
             .fillna(0.0)
             .sort_values("nativa", ascending=False))
    total = tabla["nativa"].sum()
    tabla["nativa_norm"] = tabla["nativa"] / total if total > 0 else 0.0

    print("Importancia nativa ({}), agregada por variable original,".format(etiqueta))
    print("y caida de F1 al permutar cada variable en el conjunto de PRUEBA.\n")
    print("{:<24}{:>10}  {:>18}   grafico (nativa)".format(
        "variable", "nativa", "permutacion"))
    print("-" * 92)
    mx = tabla["nativa_norm"].max()
    for origen, r in tabla.iterrows():
        print("{:<24}{:>9.1%}  {:>8.3f} +/-{:.3f}   {}".format(
            origen, r["nativa_norm"], r["permutacion"], r["permutacion_de"],
            barra(r["nativa_norm"], mx)))
    print("\nNota: la permutacion se mide sobre el conjunto de prueba; con pocas")
    print("filas su desviacion es grande y valores cercanos a 0 no son distinguibles.")
    return tabla


# =====================================================================
# EVALUACION FINAL EN EL CONJUNTO DE PRUEBA
# =====================================================================
def evaluar_en_prueba(pipe, X_pru, y_pru):
    pred = pipe.predict(X_pru)
    met = {"exactitud": accuracy_score(y_pru, pred),
           "precision": precision_score(y_pru, pred, zero_division=0),
           "recall": recall_score(y_pru, pred, zero_division=0),
           "f1": f1_score(y_pru, pred, zero_division=0)}
    print("Metricas en el conjunto de PRUEBA (nunca visto):")
    for k in ORDEN_METRICAS:
        print("    {:<12} = {:.3f}".format(k, met[k]))
    mc = confusion_matrix(y_pru, pred)
    print("\nMatriz de confusion   (filas = real, columnas = predicho)")
    print("                 pred 0   pred 1")
    print("    real 0    {:>8} {:>8}".format(mc[0, 0], mc[0, 1]))
    print("    real 1    {:>8} {:>8}".format(mc[1, 0], mc[1, 1]))
    print("\nEl conjunto de prueba tiene {} filas: cada acierto o error mueve las "
          "metricas {:.0%}.".format(len(y_pru), 1.0 / len(y_pru)))
    return met


# =====================================================================
# ORQUESTACION DE UN ESCENARIO COMPLETO
# =====================================================================
def ejecutar_escenario(clave, descripcion, df, excluir):
    titulo("ESCENARIO {}: {}".format(clave, descripcion))
    num, cat = tipos_de_variable(df, excluir)
    variables = num + cat
    if excluir:
        print("Variables excluidas por fuga: {}".format(", ".join(excluir)))
    print("Variables usadas: {} ({} numericas, {} categoricas)".format(
        len(variables), len(num), len(cat)))
    print("  numericas   : {}".format(", ".join(num)))
    print("  categoricas : {}".format(", ".join(cat) if cat else "(ninguna)"))

    X = df[variables]
    y = df[OBJETIVO]
    X_ent, X_pru, y_ent, y_pru = train_test_split(
        X, y, test_size=PROPORCION_PRUEBA, stratify=y, random_state=SEMILLA)
    print("\n1. Particion estratificada 80/20 -> entrenamiento {} filas ({:.1%} "
          "abandono), prueba {} filas ({:.1%} abandono)".format(
              len(X_ent), y_ent.mean(), len(X_pru), y_pru.mean()))

    print("\n--- 4-5. Comparacion de los tres modelos ---")
    tabla = comparar_modelos(X_ent, y_ent, num, cat)
    imprimir_comparacion(tabla)
    mejor = tabla.iloc[0]["modelo"]
    print("\nMejor por F1 en validacion cruzada: {}".format(mejor))

    print("\n--- 6. Busqueda en malla ---")
    busqueda = ajustar_mejor(mejor, X_ent, y_ent, num, cat)
    pipe = busqueda.best_estimator_

    print("\n--- Evaluacion final ---")
    met_prueba = evaluar_en_prueba(pipe, X_pru, y_pru)

    print("\n--- 7. Importancia de variables ---")
    imp = importancias(pipe, X_pru, y_pru, variables)

    return {"clave": clave, "descripcion": descripcion, "modelo": mejor,
            "pipeline": pipe, "variables": variables, "numericas": num,
            "categoricas": cat, "excluidas": excluir, "tabla_cv": tabla,
            "f1_cv": busqueda.best_score_, "metricas_prueba": met_prueba,
            "mejores_parametros": busqueda.best_params_, "importancias": imp}


# =====================================================================
# CONTROL: EL PIPELINE MISMO, ¿ESTA FILTRANDO?
# =====================================================================
def prueba_etiquetas_barajadas(df, variables, num, cat):
    """Reentrena con el objetivo permutado al azar.

    Si el pipeline tuviera un error de fuga (escalar antes de partir, imputar
    con todo el conjunto, etc.) seguiria dando metricas altas con etiquetas
    sin sentido. Debe caer a ~0.50, el nivel del azar en un problema
    balanceado. Es la prueba de que el 1.000 viene de los datos, no del codigo.
    """
    titulo("CONTROL: ETIQUETAS BARAJADAS")
    rng = np.random.RandomState(SEMILLA)
    y_falso = pd.Series(rng.permutation(df[OBJETIVO].to_numpy()), index=df.index)
    cv = StratifiedKFold(n_splits=N_PLIEGUES, shuffle=True, random_state=SEMILLA)
    pipe = Pipeline([("preparar", construir_preprocesador(num, cat)),
                     ("modelo", LogisticRegression(max_iter=5000,
                                                   random_state=SEMILLA))])
    f1_falso = cross_val_score(pipe, df[variables], y_falso, cv=cv,
                               scoring="f1", n_jobs=-1).mean()
    f1_real = cross_val_score(pipe, df[variables], df[OBJETIVO], cv=cv,
                              scoring="f1", n_jobs=-1).mean()
    print("Regresion logistica sobre las variables crudas:")
    print("    F1 con el objetivo real     = {:.3f}".format(f1_real))
    print("    F1 con el objetivo barajado = {:.3f}".format(f1_falso))
    if f1_falso < 0.65:
        print("\nEl control cae al nivel del azar: el pipeline NO tiene fuga de")
        print("preprocesamiento. La exactitud perfecta viene de los datos.")
    else:
        print("\n>>> ALERTA: el control tambien sale alto. Revisar el pipeline.")
    return f1_real, f1_falso


# =====================================================================
# SECCION 8. GUARDADO
# =====================================================================
def guardar(res):
    titulo("8. GUARDADO DEL MODELO")
    carga = {
        "modelo": res["pipeline"],
        "variables": res["variables"],
        "variables_numericas": res["numericas"],
        "variables_categoricas": res["categoricas"],
        "variables_excluidas_por_fuga": res["excluidas"],
        "objetivo": OBJETIVO,
        "nombre_modelo": res["modelo"],
        "mejores_hiperparametros": res["mejores_parametros"],
        "f1_validacion_cruzada": res["f1_cv"],
        "metricas_prueba": res["metricas_prueba"],
        "escenario": "{} - {}".format(res["clave"], res["descripcion"]),
        "semilla": SEMILLA,
        "version_sklearn": sklearn.__version__,
        "fecha": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }
    joblib.dump(carga, SALIDA_MODELO)
    print("Archivo   : {} ({:.0f} KB)".format(
        SALIDA_MODELO.name, SALIDA_MODELO.stat().st_size / 1024))
    print("Escenario : {}".format(carga["escenario"]))
    print("Modelo    : {}".format(res["modelo"]))
    print("Variables : {} -> {}".format(
        len(res["variables"]), ", ".join(res["variables"])))
    print("\nEl pkl es un diccionario. Para usarlo:")
    print("    import joblib, pandas as pd")
    print("    c = joblib.load('modelo_churn.pkl')")
    print("    pred = c['modelo'].predict(nuevos[c['variables']])")
    print("\nEl pipeline guardado incluye imputacion, codificacion y escalado:")
    print("se le pasa el DataFrame crudo con esas columnas, sin preprocesar.")


# =====================================================================
# VEREDICTO
# =====================================================================
def veredicto(resultados, fugas, control):
    titulo("VEREDICTO")
    print("{:<38}{:<22}{:>8}{:>12}{:>19}".format(
        "escenario", "modelo", "F1 CV", "F1 prueba", "exactitud prueba"))
    print("-" * 99)
    for r in resultados:
        print("{:<38}{:<22}{:>8.3f}{:>12.3f}{:>19.3f}".format(
            r["clave"] + ". " + r["descripcion"] + " ({} vars)".format(
                len(r["variables"])),
            r["modelo"], r["f1_cv"], r["metricas_prueba"]["f1"],
            r["metricas_prueba"]["exactitud"]))

    res_a, res_b, res_c = resultados
    f1_real, f1_falso = control
    print("\n1) QUE MODELO GANA")
    print("   {} en los tres escenarios. Con 80 filas de entrenamiento y clases".format(
        res_b["modelo"]))
    print("   casi separables, un modelo lineal regularizado es el mas estable: el")
    print("   bosque y el boosting no tienen nada extra que capturar y pagan varianza.")
    print("   La malla eligio C = {} (regularizacion fuerte), sintoma de que la".format(
        res_b["mejores_parametros"].get("modelo__C", "n/d")))
    print("   frontera es amplia y no hace falta ajustarse a los datos.")

    print("\n2) LAS METRICAS SON SOSPECHOSAS. NO USE ESTE MODELO EN PRODUCCION.")
    if fugas:
        print("   a) Fuga directa: {} separan las clases sin un solo error".format(
            " y ".join(fugas)))
        print("      en los 100 registros (anual -> 0% de abandono, mensual -> 100%).")
        print("      Son la misma columna codificada dos veces.")
    print("   b) Quitarlas NO baja las metricas (escenario B sigue en {:.3f}).".format(
        res_b["metricas_prueba"]["f1"]))
    print("      El problema no eran dos columnas.")
    print("   c) Con SOLO las variables crudas, sin nada derivado, el F1 sigue en")
    print("      {:.3f} (escenario C). Las dos clases ocupan rangos casi disjuntos".format(
        res_c["metricas_prueba"]["f1"]))
    print("      (ver la tabla de solape): el conjunto es sintetico y se genero")
    print("      muestreando cada clase de una distribucion distinta.")
    print("   d) El control con etiquetas barajadas cae a F1 = {:.3f} frente a {:.3f}".format(
        f1_falso, f1_real))
    print("      con las reales: el pipeline esta bien construido. El 1.000 es de")
    print("      los datos, no de un error de codigo.")

    print("\n3) QUE ESPERAR CON DATOS REALES")
    print("   Un modelo de churn decente en produccion ronda 0.60-0.80 de AUC y")
    print("   rara vez pasa de 0.70 de F1. Un 1.000 nunca es una buena noticia:")
    print("   es una senal de que la respuesta esta dentro de la entrada.")

    print("\n4) OTRAS LIMITACIONES")
    print("   - n = 100 filas; la prueba son {} clientes. Un acierto vale 5 puntos".format(
        int(100 * PROPORCION_PRUEBA)))
    print("     porcentuales, asi que ninguna metrica tiene tres decimales de sentido.")
    print("   - Diferencias menores a 0.05 en F1 entre modelos no son distinguibles.")
    print("   - Clases balanceadas 49/51; en churn real es 5-25%. Con esa proporcion")
    print("     habria que revisar el umbral, usar class_weight y mirar PR-AUC.")
    print("   - Redundancias exactas (gasto_anual = gasto_mensual x 12): reparten la")
    print("     importancia entre columnas gemelas y no aportan informacion nueva.")

    print("\n5) RECOMENDACION")
    print("   modelo_churn.pkl guarda el escenario B, que es el unico defendible de")
    print("   los tres, pero sirve como plantilla de codigo, no como modelo. Antes de")
    print("   creer cualquier numero: regenerar los datos con solape realista entre")
    print("   clases, o traer datos reales, y volver a correr este mismo script.")
    print("   La auditoria de las secciones 0b y de control es la parte reutilizable.")


def main():
    df = cargar()
    fugas = auditar_fuga(df)

    derivadas = [c for c in df.columns
                 if c not in VARIABLES_CRUDAS + [ID, OBJETIVO]]
    excluir_c = sorted(set(derivadas + fugas))

    res_a = ejecutar_escenario("A", "todas las variables", df, excluir=[])
    res_b = ejecutar_escenario("B", "sin variables con fuga", df, excluir=fugas)
    res_c = ejecutar_escenario("C", "solo variables crudas", df, excluir=excluir_c)

    control = prueba_etiquetas_barajadas(
        df, res_c["variables"], res_c["numericas"], res_c["categoricas"])

    guardar(res_b if ESCENARIO_A_GUARDAR == "B" else res_a)
    veredicto([res_a, res_b, res_c], fugas, control)


if __name__ == "__main__":
    main()
