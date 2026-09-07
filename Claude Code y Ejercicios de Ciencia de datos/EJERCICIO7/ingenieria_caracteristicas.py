# -*- coding: utf-8 -*-
"""
Ingenieria de caracteristicas para prediccion de abandono (churn).

Entrada : ../churn_clientes_crudo.csv
Salida  : churn_clientes_features.csv

DISENO EXTENSIBLE
-----------------
Cada variable nueva se declara con el decorador @caracteristica(...).
Para agregar mas variables basta con pegar un bloque nuevo en la
SECCION 3 (no hay que tocar nada mas): el resto del script las recoge
automaticamente para construirlas, correlacionarlas y auditarlas.
"""

from pathlib import Path

import numpy as np
import pandas as pd

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 50)

BASE = Path(__file__).resolve().parent
NOMBRE_ENTRADA = "churn_clientes_crudo.csv"
SALIDA = BASE / "churn_clientes_features.csv"
SALIDA_COMPLETA = BASE / "churn_clientes_features_completo.csv"
OBJETIVO = "abandono"


def localizar_entrada() -> Path:
    """Busca el csv crudo en la carpeta del script, la de arriba y ../../datos."""
    candidatos = [BASE / NOMBRE_ENTRADA,
                  BASE.parent / NOMBRE_ENTRADA,
                  BASE.parent.parent / "datos" / NOMBRE_ENTRADA]
    for c in candidatos:
        if c.exists():
            return c
    rutas = "\n  ".join(str(c) for c in candidatos)
    raise FileNotFoundError(f"No encuentro {NOMBRE_ENTRADA} en:\n  {rutas}")


ENTRADA = localizar_entrada()


# =====================================================================
# SECCION 1. CARGA Y ANALISIS DE LAS VARIABLES EXISTENTES
# =====================================================================
def cargar() -> pd.DataFrame:
    df = pd.read_csv(ENTRADA)
    print("=" * 78)
    print("1. ANALISIS DEL CONJUNTO CRUDO")
    print("=" * 78)
    print(f"Filas: {len(df)}   Columnas: {df.shape[1]}   "
          f"IDs duplicados: {df['id_cliente'].duplicated().sum()}   "
          f"Nulos totales: {int(df.isna().sum().sum())}")
    print(f"Balance del objetivo: {df[OBJETIVO].mean():.1%} de abandono "
          f"({df[OBJETIVO].sum()} de {len(df)})\n")

    num = df.select_dtypes(include=np.number).columns.drop(OBJETIVO)
    print("-- Descriptivos numericos --")
    print(df[num].describe().T.round(2), "\n")

    print("-- Perfil medio por clase (0 = se queda, 1 = abandona) --")
    print(df.groupby(OBJETIVO)[num].mean().T.round(2), "\n")

    cats = df.select_dtypes(exclude=np.number).columns.drop("id_cliente")
    print("-- Tasa de abandono por variable categorica --")
    for c in cats:
        t = df.groupby(c)[OBJETIVO].agg(n="count", tasa="mean")
        t["tasa"] = (t["tasa"] * 100).round(1).astype(str) + "%"
        print(f"\n[{c}]\n{t}")

    print("\n-- Correlaciones entre predictores numericos (|r| >= 0.30) --")
    corr = df[num].corr()
    vistos = set()
    for a in num:
        for b in num:
            if a != b and (b, a) not in vistos and abs(corr.loc[a, b]) >= 0.30:
                vistos.add((a, b))
                print(f"  {a:<20} ~ {b:<20} r = {corr.loc[a, b]:+.3f}")
    print()
    return df


# =====================================================================
# SECCION 2. REGISTRO DE CARACTERISTICAS (motor extensible)
# =====================================================================
REGISTRO: list[dict] = []


def caracteristica(nombre: str, familia: str, porque: str):
    """Registra una variable nueva. La funcion recibe el DataFrame y
    devuelve una Serie."""
    def envoltura(fn):
        REGISTRO.append({"nombre": nombre, "familia": familia,
                         "porque": porque, "fn": fn})
        return fn
    return envoltura


def construir(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for f in REGISTRO:
        out[f["nombre"]] = f["fn"](out)
    return out


# =====================================================================
# SECCION 3. DEFINICION DE LAS VARIABLES NUEVAS
#            <<< AGREGAR AQUI LAS PROXIMAS VARIABLES >>>
# =====================================================================
EPS = 1e-9  # evita division por cero


# ---- Familia: RAZONES -----------------------------------------------
@caracteristica(
    "gasto_por_producto", "razon",
    "Mide cuanto vale cada producto contratado. Un cliente con varios "
    "productos pero gasto bajo tiene una relacion superficial y barata de "
    "abandonar; gasto alto por producto implica dependencia del servicio.")
def _f01(d):
    return (d.gasto_mensual / (d.num_productos + EPS)).round(2)


@caracteristica(
    "valor_vida_estimado", "razon",
    "gasto_mensual x meses_antiguedad: proxy del valor acumulado (CLV). "
    "Resume en un solo numero cuanto ha invertido el cliente en la "
    "relacion; el costo hundido desincentiva la baja.")
def _f02(d):
    return (d.gasto_mensual * d.meses_antiguedad).round(2)


@caracteristica(
    "gasto_relativo_region", "razon",
    "Gasto del cliente dividido entre la mediana de su region. Normaliza "
    "por poder adquisitivo local: gastar 60 es poco en una region cara y "
    "mucho en una barata, y el nivel relativo al grupo de pares suele "
    "predecir mejor que el nivel absoluto.")
def _f03(d):
    med = d.groupby("region")["gasto_mensual"].transform("median")
    return (d.gasto_mensual / (med + EPS)).round(3)


@caracteristica(
    "densidad_adopcion", "razon",
    "Productos contratados por unidad de tiempo (log) de antiguedad. "
    "Distingue al cliente que adopto rapido (comprometido) del que lleva "
    "anios con un solo producto (vinculo debil).")
def _f04(d):
    return (d.num_productos / np.log1p(d.meses_antiguedad)).round(3)


# ---- Familia: TASAS --------------------------------------------------
@caracteristica(
    "tickets_por_mes", "tasa",
    "Tickets de soporte normalizados por antiguedad. 5 tickets en 2 meses "
    "es friccion severa; 5 en 5 anios es ruido. La intensidad de la "
    "friccion, no su conteo bruto, es lo que antecede a la baja.")
def _f05(d):
    return (d.tickets_soporte / (d.meses_antiguedad + EPS)).round(3)


@caracteristica(
    "ratio_inactividad", "tasa",
    "dias_ultimo_acceso dividido entre la duracion total de la relacion "
    "(en dias). Un silencio de 30 dias en un cliente de 1 mes es un "
    "abandono practico; en uno de 5 anios es una pausa. Captura el "
    "desenganche relativo a su propio historial.")
def _f06(d):
    return (d.dias_ultimo_acceso / (d.meses_antiguedad * 30.44 + EPS)).round(3)


@caracteristica(
    "costo_soporte_por_ingreso", "tasa",
    "Tickets acumulados por cada 100 unidades de gasto mensual. Aisla al "
    "cliente caro de atender y poco rentable, el perfil que mas rota y el "
    "que menos conviene retener con descuentos.")
def _f07(d):
    return (100 * d.tickets_soporte / (d.gasto_mensual + EPS)).round(3)


# ---- Familia: BANDERAS DE COMPORTAMIENTO -----------------------------
@caracteristica(
    "bandera_inactivo_30d", "bandera",
    "1 si no accede hace mas de 30 dias. Umbral operativo clasico: el "
    "desuso precede a la cancelacion. Da al modelo un corte no lineal que "
    "una variable continua obliga a aproximar.")
def _f08(d):
    return (d.dias_ultimo_acceso > 30).astype(int)


@caracteristica(
    "bandera_cliente_nuevo", "bandera",
    "1 si tiene 6 meses o menos. El riesgo de baja no es lineal en el "
    "tiempo: se concentra en el periodo de onboarding, antes de que el "
    "habito se consolide.")
def _f09(d):
    return (d.meses_antiguedad <= 6).astype(int)


@caracteristica(
    "bandera_soporte_intensivo", "bandera",
    "1 si sus tickets superan el percentil 75 de la cartera. Marca la cola "
    "de clientes frustrados, donde se concentra buena parte del churn "
    "evitable.")
def _f10(d):
    return (d.tickets_soporte >= d.tickets_soporte.quantile(0.75)).astype(int)


@caracteristica(
    "bandera_pago_manual", "bandera",
    "1 si paga por transferencia o paypal (metodos que exigen una accion "
    "activa cada ciclo). La domiciliacion y la tarjeta renuevan solas: la "
    "friccion de pago es una via directa a la baja pasiva.")
def _f11(d):
    return d.metodo_pago.isin(["transferencia", "paypal"]).astype(int)


@caracteristica(
    "contrato_mensual", "bandera",
    "Codificacion binaria del tipo de contrato. Sin permanencia el cliente "
    "puede irse cualquier mes: es la barrera de salida mas importante. "
    "OJO: revisar la auditoria de fuga al final.")
def _f12(d):
    return (d.tipo_contrato == "mensual").astype(int)


# ---- Familia: AGRUPAMIENTOS -----------------------------------------
@caracteristica(
    "grupo_antiguedad", "agrupamiento",
    "Tramos del ciclo de vida (0-6 / 7-12 / 13-24 / 25+ meses) como ordinal "
    "0-3. Convierte una relacion escalonada en algo que los modelos "
    "lineales capturan sin transformaciones.")
def _f13(d):
    return pd.cut(d.meses_antiguedad, [-1, 6, 12, 24, np.inf],
                  labels=[0, 1, 2, 3]).astype(int)


@caracteristica(
    "grupo_valor", "agrupamiento",
    "Cuartil de gasto mensual (0-3). Segmenta la cartera en niveles de "
    "valor comparables y permite leer el churn por segmento comercial, "
    "que es como se decide el presupuesto de retencion.")
def _f14(d):
    return pd.qcut(d.gasto_mensual, 4, labels=[0, 1, 2, 3]).astype(int)


@caracteristica(
    "indice_riesgo_conductual", "agrupamiento",
    "Suma de tres senales conductuales (inactivo + soporte intensivo + "
    "cliente nuevo), de 0 a 3. Score compacto que captura la acumulacion "
    "de riesgo: dos senales debiles juntas pesan mas que una fuerte "
    "aislada.")
def _f15(d):
    return (d.bandera_inactivo_30d + d.bandera_soporte_intensivo
            + d.bandera_cliente_nuevo)


# ---- Familia: RESIDUALES --------------------------------------------
@caracteristica(
    "gasto_inesperado", "residual",
    "Residual de la regresion gasto_mensual ~ meses_antiguedad: cuanto "
    "gasta el cliente por encima (+) o por debajo (-) de lo que su "
    "antiguedad predice. El nivel de gasto ya lo capturan otras variables; "
    "lo que este residual aisla es la DESVIACION respecto de su cohorte "
    "temporal, que es la firma de una degradacion silenciosa: un veterano "
    "que gasta como un cliente nuevo ya se esta yendo aunque siga activo. "
    "No usa el objetivo, asi que no introduce fuga.")
def _f16(d):
    b, a = np.polyfit(d.meses_antiguedad, d.gasto_mensual, 1)
    return (d.gasto_mensual - (a + b * d.meses_antiguedad)).round(2)


# ---- Familia: RELATIVAS A COHORTE ------------------------------------
@caracteristica(
    "percentil_inactividad_cohorte", "cohorte",
    "Percentil (0-1) de dias_ultimo_acceso DENTRO de su tramo de "
    "antiguedad. Los dias absolutos confunden dos cosas: que un cliente "
    "nuevo acceda poco es normal, que un veterano lo haga es alarma. Al "
    "rankear contra sus pares de la misma etapa del ciclo de vida, la "
    "variable responde 'esta mas desenganchado que los suyos' en lugar de "
    "'cuantos dias lleva sin entrar'. Ademas es robusta a valores extremos "
    "por tratarse de un rango.")
def _f17(d):
    return (d.groupby("grupo_antiguedad")["dias_ultimo_acceso"]
            .rank(pct=True).round(3))


# ---- Familia: INTERACCIONES ------------------------------------------
@caracteristica(
    "interaccion_edad_friccion", "interaccion",
    "Edad estandarizada multiplicada por bandera_pago_manual (0 si el pago "
    "es automatico). Hipotesis: la edad sola no predice nada (r = -0.03 en "
    "el crudo), pero podria importar CONDICIONADA a que el pago exija una "
    "accion cada ciclo, donde la afinidad digital se vuelve relevante. Es "
    "una prueba explicita de si una variable inutil se rescata en "
    "interaccion. Si sale plana el diagnostico es firme: la edad no aporta "
    "ni sola ni acompanada, y se puede descartar.")
def _f18(d):
    edad_z = (d.edad - d.edad.mean()) / d.edad.std(ddof=0)
    return (edad_z * d.bandera_pago_manual).round(3)


# =====================================================================
# SECCION 4. CORRELACION CON EL OBJETIVO Y AUDITORIA DE FUGA
# =====================================================================
def auc_univariada(x: pd.Series, y: pd.Series) -> float:
    """AUC de una sola variable via el estadistico de Mann-Whitney."""
    r = x.rank()
    n1, n0 = int(y.sum()), int((1 - y).sum())
    if n1 == 0 or n0 == 0:
        return np.nan
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def _envolver(texto: str, ancho: int) -> list[str]:
    palabras, linea, salida = texto.split(), "", []
    for p in palabras:
        if len(linea) + len(p) + 1 > ancho:
            salida.append(linea)
            linea = p
        else:
            linea = f"{linea} {p}".strip()
    if linea:
        salida.append(linea)
    return salida


def reportar(df: pd.DataFrame) -> pd.DataFrame:
    print("=" * 78)
    print("2-3. VARIABLES NUEVAS Y POR QUE DEBERIAN PREDECIR")
    print("=" * 78)
    for i, f in enumerate(REGISTRO, 1):
        print(f"\n{i:>2}. {f['nombre']}  [{f['familia']}]")
        for linea in _envolver(f["porque"], 72):
            print(f"    {linea}")

    nuevas = {f["nombre"] for f in REGISTRO}
    numericas = [c for c in df.select_dtypes(include=np.number).columns
                 if c != OBJETIVO]
    y = df[OBJETIVO]

    filas = []
    for c in numericas:
        s = df[c].astype(float)
        filas.append({
            "variable": c,
            "origen": "NUEVA" if c in nuevas else "original",
            "corr_pearson": s.corr(y),
            "corr_spearman": s.rank().corr(y.rank()),  # Spearman sin scipy
            "auc": auc_univariada(s, y),
        })
    tabla = pd.DataFrame(filas)
    tabla["abs_r"] = tabla.corr_pearson.abs()
    tabla = tabla.sort_values("abs_r", ascending=False).reset_index(drop=True)

    print("\n" + "=" * 78)
    print("4. CORRELACION DE CADA VARIABLE CON 'abandono'")
    print("   (AUC = poder discriminante univariado; 0.5 = azar, 1.0 = perfecto)")
    print("=" * 78)
    print(tabla[["variable", "origen", "corr_pearson", "corr_spearman", "auc"]]
          .to_string(index=False,
                     formatters={"corr_pearson": "{:+.3f}".format,
                                 "corr_spearman": "{:+.3f}".format,
                                 "auc": "{:.3f}".format}))

    print("\n-- Categoricas: tasa de abandono por nivel --")
    for c in ["tipo_contrato", "metodo_pago", "region"]:
        t = df.groupby(c)[OBJETIVO].mean().round(3).to_dict()
        rango = max(t.values()) - min(t.values())
        print(f"  {c:<15} {t}   (rango = {rango:.3f})")

    print("\n" + "=" * 78)
    print("AUDITORIA DE POSIBLE FUGA DE INFORMACION")
    print("=" * 78)
    alertas = []
    for _, r in tabla.iterrows():
        if r["abs_r"] >= 0.95 or r["auc"] >= 0.99 or r["auc"] <= 0.01:
            nivel = "CRITICO"
        elif r["abs_r"] >= 0.80 or r["auc"] >= 0.95 or r["auc"] <= 0.05:
            nivel = "ALTO"
        else:
            continue
        alertas.append((nivel, r["variable"], r["corr_pearson"], r["auc"]))

    for c in ["tipo_contrato", "metodo_pago", "region"]:
        t = df.groupby(c)[OBJETIVO].mean()
        if bool(((t == 0) | (t == 1)).all()):
            alertas.append(("CRITICO", f"{c} (separacion perfecta)",
                            np.nan, np.nan))

    if not alertas:
        print("  Sin senales de fuga.")
    for nivel, var, r, a in alertas:
        r_txt = "  n/a " if pd.isna(r) else f"{r:+.3f}"
        a_txt = " n/a " if pd.isna(a) else f"{a:.3f}"
        print(f"  [{nivel:<8}] {var:<34} r = {r_txt}   AUC = {a_txt}")
    return tabla


# =====================================================================
# SECCION 5. PODA: QUE SE DESCARTA Y POR QUE
#            <<< AGREGAR AQUI LOS PROXIMOS DESCARTES >>>
#
# El pipeline construye TODO primero y poda despues: solo asi hay
# evidencia (correlacion, AUC, redundancia) para justificar cada corte.
# =====================================================================
DESCARTES: list[tuple[str, str, str]] = [
    # ---- Fuga de informacion: obligatorio, no es opcional -------------
    ("tipo_contrato", "FUGA",
     "Separacion perfecta: anual = 0% de bajas, mensual = 100%. O el campo "
     "se actualiza al cancelar, o el dataset se genero desde el. En "
     "cualquier caso el modelo aprenderia un if, no un patron."),
    ("contrato_mensual", "FUGA",
     "Codificacion binaria de lo anterior. r = +1.000 exacto. Misma causa, "
     "mismo destino."),

    # ---- Sin sentido de negocio: la magnitud no se puede leer ---------
    ("densidad_adopcion", "SIN LECTURA",
     "Productos por unidad logaritmica de antiguedad. Nadie en retencion "
     "puede interpretar '0.497 productos por log-mes' ni accionar sobre "
     "ello. Ademas es la mas debil de su familia (AUC 0.334). Una variable "
     "que no se puede explicar en una junta no sobrevive a la junta."),
    ("interaccion_edad_friccion", "HIPOTESIS REFUTADA",
     "Se construyo para probar si la edad se rescata condicionada a la "
     "friccion de pago. AUC 0.460, r = -0.025: no. La prueba cumplio su "
     "funcion y el resultado es descartarla."),

    # ---- Sin senal: la hipotesis de negocio era razonable, los datos no
    ("gasto_por_producto", "SIN SENAL",
     "AUC 0.525, indistinguible del azar. Gasto y numero de productos "
     "crecen juntos (r = +0.58), asi que el cociente se cancela y no queda "
     "informacion util."),
    ("bandera_pago_manual", "SIN SENAL",
     "AUC 0.441: el signo va en direccion CONTRARIA a la hipotesis (los de "
     "pago manual abandonan algo menos). En esta cartera el metodo de pago "
     "no mueve la aguja (rango de tasas de solo 19 puntos)."),
    ("percentil_inactividad_cohorte", "SIN SENAL",
     "AUC 0.553. El ranking dentro del tramo de antiguedad no aporta "
     "porque el propio tramo ya casi determina el resultado: dentro de "
     "cada cohorte apenas queda varianza que explicar."),
    ("edad", "RUIDO CONFIRMADO",
     "Variable original, r = -0.025. Dos intentos de rescate (interaccion "
     "con friccion de pago, y su uso implicito en agrupamientos) fallaron. "
     "Se descarta con evidencia, no por sospecha."),

    # ---- Redundantes: miden algo que ya esta en el conjunto -----------
    ("gasto_relativo_region", "REDUNDANTE",
     "r = +0.987 con gasto_mensual. La normalizacion regional era una buena "
     "idea, pero aqui las regiones tienen medianas casi iguales, asi que la "
     "variable es una copia reescalada del gasto."),
    ("grupo_valor", "REDUNDANTE",
     "r = +0.924 con gasto_mensual y con menos informacion (4 niveles en "
     "vez de continuo). Sigue siendo util para REPORTAR por segmento "
     "comercial, pero no para entrenar."),
]

MOTIVOS_ORDEN = ["FUGA", "SIN LECTURA", "HIPOTESIS REFUTADA", "SIN SENAL",
                 "RUIDO CONFIRMADO", "REDUNDANTE"]


def podar(df: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 78)
    print("5. PODA: VARIABLES DESCARTADAS")
    print("=" * 78)
    fuera = []
    for motivo in MOTIVOS_ORDEN:
        grupo = [d for d in DESCARTES if d[1] == motivo]
        if not grupo:
            continue
        print(f"\n--- {motivo} ---")
        for nombre, _, razon in grupo:
            if nombre not in df.columns:
                print(f"  ! {nombre}: no esta en el conjunto, se ignora")
                continue
            fuera.append(nombre)
            print(f"  - {nombre}")
            for linea in _envolver(razon, 70):
                print(f"      {linea}")
    return df.drop(columns=fuera)


# =====================================================================
# SECCION 6. GUARDADO
# =====================================================================
def guardar(completo: pd.DataFrame, curado: pd.DataFrame) -> None:
    for nombre, d in ((SALIDA_COMPLETA, completo), (SALIDA, curado)):
        orden = [c for c in d.columns if c != OBJETIVO] + [OBJETIVO]
        d[orden].to_csv(nombre, index=False)

    nuevas_vivas = sum(1 for f in REGISTRO if f["nombre"] in curado.columns)
    print("\n" + "=" * 78)
    print("6. GUARDADO")
    print("=" * 78)
    print(f"  {SALIDA_COMPLETA.name:<34} {completo.shape[0]} x "
          f"{completo.shape[1]}   (auditoria: incluye lo descartado)")
    print(f"  {SALIDA.name:<34} {curado.shape[0]} x {curado.shape[1]}   "
          f"(para modelar: {nuevas_vivas} variables nuevas que sobreviven)")
    print(f"\n  Descartadas: {len(DESCARTES)}   "
          f"Conservadas: {curado.shape[1] - 2} predictores + id + objetivo")
    print("=" * 78)


if __name__ == "__main__":
    crudo = cargar()
    enriquecido = construir(crudo)
    reportar(enriquecido)
    curado = podar(enriquecido)
    guardar(enriquecido, curado)
