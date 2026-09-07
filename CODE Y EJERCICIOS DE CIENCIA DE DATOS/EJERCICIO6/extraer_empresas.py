"""
Extraccion y analisis de la tabla "List of largest companies by revenue" (Wikipedia).

Salidas:
  - empresas_por_ingreso.csv      datos limpios
  - resumen_por_pais.csv          agregado por pais
  - resumen_por_industria.csv     agregado por industria principal
  - top15_ingresos.png            grafica de barras horizontal

Solo usa la biblioteca estandar + pandas/matplotlib/lxml (no requiere requests ni bs4).
"""

import io
import re
import sys
import urllib.request
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

URL = "https://en.wikipedia.org/wiki/List_of_largest_companies_by_revenue"
UA = "Mozilla/5.0 (compatible; ejercicio-academico-pandas/1.0)"
SALIDA = Path(__file__).parent


# --------------------------------------------------------------------------
# 1. Descarga y lectura de la tabla principal
# --------------------------------------------------------------------------
def descargar_html(url: str = URL) -> str:
    """Descarga el HTML. Wikipedia rechaza el User-Agent por defecto de urllib."""
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read().decode("utf-8")


def tabla_principal(html: str) -> pd.DataFrame:
    """
    Devuelve la primera tabla que tiene las columnas del ranking.

    read_html expande rowspan/colspan, algo indispensable aqui: las columnas de
    pais e industria usan celdas combinadas en varias filas y un parseo manual
    del HTML dejaria los valores recorridos.
    """
    tablas = pd.read_html(io.StringIO(html))
    for t in tablas:
        cols = " ".join(str(c) for c in t.columns).lower()
        if "revenue" in cols and "employees" in cols:
            return t
    raise RuntimeError("No se encontro la tabla principal en la pagina.")


# --------------------------------------------------------------------------
# 2. Limpieza de formato (comas, moneda, notas al pie)
# --------------------------------------------------------------------------
NOTAS = re.compile(r"\[[^\]]*\]")          # [5], [note 1], [a]
MONEDA = re.compile(r"US\$|USD|[$€£¥]")
NO_NUMERICO = re.compile(r"[^\d.\-]")      # lo que sobra tras quitar moneda/comas
FALTANTES = {"", "n/a", "na", "nan", "none", "-", "--", "—", "–", "?"}


def limpiar_texto(valor) -> str:
    """Quita notas al pie, espacios duros y espacios redundantes."""
    if pd.isna(valor):
        return ""
    s = str(valor).replace("\xa0", " ").replace("​", "")
    s = NOTAS.sub("", s)                   # notas al pie: [5], [note 1]
    return re.sub(r"\s+", " ", s).strip()


def a_numero(valor) -> float:
    """
    Convierte texto a float tolerando:
      comas de miles ("1,576,000"), simbolos de moneda ("$716", "US$"),
      notas al pie ("716[note 2]"), signo menos Unicode ("-0.4"),
      negativos entre parentesis ("(1.2)") y marcadores de faltante ("n/a").
    """
    s = limpiar_texto(valor)
    if s.lower() in FALTANTES:
        return float("nan")

    negativo = s.startswith("(") and s.endswith(")")
    s = s.strip("()")
    s = s.replace("−", "-").replace("–", "-")   # menos Unicode / guion corto
    s = MONEDA.sub("", s)
    s = s.replace(",", "").replace(" ", "")
    s = NO_NUMERICO.sub("", s)                            # restos: "%", "approx.", etc.
    s = re.sub(r"(?<=.)-", "", s)                         # menos solo valido al inicio

    if s in ("", "-", "."):
        return float("nan")
    try:
        numero = float(s)
    except ValueError:
        return float("nan")
    return -numero if negativo else numero


# Los nombres de industria empiezan con mayuscula y siguen en minuscula
# ("Oil and gas"), asi que un salto minuscula->mayuscula marca la union de dos
# industrias en la misma celda: "Retail Information technology".
CORTE_INDUSTRIA = re.compile(r"(?<=[a-z]) (?=[A-Z])")


def normalizar(df: pd.DataFrame) -> pd.DataFrame:
    """Renombra columnas, limpia texto y castea numeros."""
    # Encabezado de dos niveles -> un solo nivel, sin notas al pie
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [limpiar_texto(c[0]) for c in df.columns]
    else:
        df.columns = [limpiar_texto(c) for c in df.columns]

    mapa = {
        "Ranks": "posicion", "Rank": "posicion",
        "Name": "empresa",
        "Industry": "industria",
        "Revenue": "ingresos_miles_millones_usd",
        "Profit": "utilidad_miles_millones_usd",
        "Employees": "empleados",
        "Headquarters": "pais",
        "State-owned": "estatal",
    }
    df = df.rename(columns=mapa)
    df = df.drop(columns=[c for c in ("Ref.", "Ref") if c in df.columns])

    for col in ("empresa", "industria", "pais"):
        df[col] = df[col].map(limpiar_texto)

    for col in ("posicion", "ingresos_miles_millones_usd",
                "utilidad_miles_millones_usd", "empleados"):
        df[col] = df[col].map(a_numero)

    df["posicion"] = df["posicion"].astype("Int64")
    df["empleados"] = df["empleados"].round().astype("Int64")
    df["estatal"] = df["estatal"].map(limpiar_texto).str.lower().isin(
        {"yes", "y", "si", "sí", "true"}
    )

    # Una celda puede listar varias industrias (p. ej. Amazon)
    df["industria"] = df["industria"].map(
        lambda s: "; ".join(CORTE_INDUSTRIA.split(s)) if s else s
    )
    df["industria_principal"] = df["industria"].str.split(";").str[0].str.strip()

    df = df.dropna(subset=["empresa", "ingresos_miles_millones_usd"])
    df = df[df["empresa"] != ""]

    orden = ["posicion", "empresa", "ingresos_miles_millones_usd",
             "utilidad_miles_millones_usd", "empleados", "industria",
             "industria_principal", "pais", "estatal"]
    return df[orden].sort_values("posicion").reset_index(drop=True)


# --------------------------------------------------------------------------
# 3. Grafica
# --------------------------------------------------------------------------
SURFACE = "#fcfcfb"
TINTA = "#0b0b0b"
TINTA_SEC = "#52514e"
SERIE = "#2a78d6"


def grafica_top15(df: pd.DataFrame, ruta: Path) -> None:
    top = df.nlargest(15, "ingresos_miles_millones_usd").iloc[::-1]  # mayor arriba

    fig, ax = plt.subplots(figsize=(10, 7), dpi=160)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    ax.barh(top["empresa"], top["ingresos_miles_millones_usd"],
            height=0.62, color=SERIE, zorder=3)

    # Etiqueta directa al final de cada barra: no hace falta volver al eje
    maximo = float(top["ingresos_miles_millones_usd"].max())
    for y, v in enumerate(top["ingresos_miles_millones_usd"]):
        ax.text(v + maximo * 0.012, y, f"{v:,.0f}", va="center",
                fontsize=9, color=TINTA_SEC, zorder=4)

    ax.set_xlim(0, maximo * 1.12)
    ax.xaxis.grid(True, color="#e4e3df", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.yaxis.grid(False)
    for lado in ("top", "right", "left"):
        ax.spines[lado].set_visible(False)
    ax.spines["bottom"].set_color("#d8d7d2")
    ax.tick_params(axis="both", length=0, colors=TINTA_SEC, labelsize=9.5)
    ax.set_xlabel("Ingresos anuales (miles de millones de USD)",
                  fontsize=9.5, color=TINTA_SEC, labelpad=10)

    # Titulo, bajada y fuente alineados al borde izquierdo de la figura
    fig.text(0.02, 0.955, "Las 15 empresas con mayores ingresos del mundo",
             fontsize=15, color=TINTA, ha="left", va="center", weight="bold")
    fig.text(0.02, 0.912, "Ingresos anuales reportados, en miles de millones de USD",
             fontsize=9.5, color=TINTA_SEC, ha="left", va="center")
    fig.text(0.02, 0.022,
             "Fuente: Wikipedia, «List of largest companies by revenue»",
             fontsize=8, color=TINTA_SEC, ha="left", va="center")

    fig.tight_layout(rect=[0.01, 0.045, 0.99, 0.885])
    fig.savefig(ruta, facecolor=SURFACE)
    plt.close(fig)


# --------------------------------------------------------------------------
# 4. Flujo principal
# --------------------------------------------------------------------------
def main() -> int:
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)

    print("Descargando la pagina de Wikipedia...")
    try:
        html = descargar_html()
    except Exception as e:                                  # red caida, 403, etc.
        print(f"ERROR: no se pudo descargar la pagina ({e})", file=sys.stderr)
        return 1

    # ---- 1. DataFrame -----------------------------------------------------
    df = normalizar(tabla_principal(html))
    print(f"\n[1] DataFrame cargado: {df.shape[0]} filas x {df.shape[1]} columnas")
    print(df.dtypes.to_string())

    # ---- 2. Top 10 --------------------------------------------------------
    top10 = df.nlargest(10, "ingresos_miles_millones_usd")
    print("\n[2] TOP 10 EMPRESAS POR INGRESOS (miles de millones de USD)")
    print(top10[["posicion", "empresa", "ingresos_miles_millones_usd",
                 "utilidad_miles_millones_usd", "empleados",
                 "industria_principal", "pais"]].to_string(index=False))

    # ---- 3. Grafica -------------------------------------------------------
    ruta_png = SALIDA / "top15_ingresos.png"
    grafica_top15(df, ruta_png)
    print(f"\n[3] Grafica guardada en: {ruta_png.name}")

    # ---- 4. Agrupaciones --------------------------------------------------
    agg = {
        "empresas": ("empresa", "count"),
        "ingresos_totales": ("ingresos_miles_millones_usd", "sum"),
        "ingresos_promedio": ("ingresos_miles_millones_usd", "mean"),
        "utilidad_total": ("utilidad_miles_millones_usd", "sum"),
        "empleados_totales": ("empleados", "sum"),
    }
    por_pais = (df.groupby("pais").agg(**agg)
                  .sort_values("ingresos_totales", ascending=False).round(1))
    por_industria = (df.groupby("industria_principal").agg(**agg)
                       .sort_values("ingresos_totales", ascending=False).round(1))

    print("\n[4a] AGRUPADO POR PAIS")
    print(por_pais.to_string())
    print("\n[4b] AGRUPADO POR INDUSTRIA")
    print(por_industria.to_string())

    # ---- 5. Guardar -------------------------------------------------------
    ruta_csv = SALIDA / "empresas_por_ingreso.csv"
    df.to_csv(ruta_csv, index=False, encoding="utf-8-sig")
    por_pais.to_csv(SALIDA / "resumen_por_pais.csv", encoding="utf-8-sig")
    por_industria.to_csv(SALIDA / "resumen_por_industria.csv", encoding="utf-8-sig")
    print(f"\n[5] Datos guardados en: {ruta_csv.name}, "
          "resumen_por_pais.csv, resumen_por_industria.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
