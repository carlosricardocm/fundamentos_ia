# -*- coding: utf-8 -*-
"""
Descarga las 20 criptomonedas más grandes por capitalización de mercado (USD)
desde la API pública de CoinGecko, las carga en un DataFrame de pandas,
grafica la capitalización y guarda todo en datos_cripto.csv.

Solo biblioteca estándar + pandas + matplotlib (no requiere `requests`).

Uso:
    python cripto_coingecko.py

Opcional: si tienes una llave Demo de CoinGecko, configúrala antes de correr
    setx COINGECKO_API_KEY "CG-xxxxxxxx"     (Windows; reabre la terminal)
y el script la enviará en el encabezado x-cg-demo-api-key (30 llamadas/min
en vez de las pocas llamadas/min del acceso público anónimo).
"""
import json
import os
import ssl
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from matplotlib.ticker import FuncFormatter

# ================================================================ 1. PETICIÓN
BASE = "https://api.coingecko.com/api/v3"
ENDPOINT = "/coins/markets"

# --- Parámetros de la petición (se imprimen en pantalla al ejecutar) --------
PARAMS = {
    "vs_currency": "usd",                        # todo valuado en dólares
    "order": "market_cap_desc",                  # ordenado por cap. de mercado
    "per_page": 20,                              # las 20 más grandes
    "page": 1,                                   # primera (y única) página
    "sparkline": "false",                        # sin la serie de 7 días (pesa mucho)
    "price_change_percentage": "1h,24h,7d,30d",  # variaciones extra
    "locale": "es",                              # nombres localizados
    "precision": "full",                         # sin redondeo del precio
}

TIEMPO_ESPERA = 20        # segundos de timeout por intento
MAX_INTENTOS = 5          # reintentos ante 429 / 5xx / fallas de red
ESPERA_BASE = 2.0         # segundos; backoff exponencial 2, 4, 8, 16...

SALIDA_CSV = "datos_cripto.csv"
SALIDA_PNG = "capitalizacion_mercado.png"


def encabezados():
    h = {
        "Accept": "application/json",
        # CoinGecko responde 403 al User-Agent por defecto de urllib.
        "User-Agent": "clase-iimas-ciencia-de-datos/1.0",
    }
    llave = os.environ.get("COINGECKO_API_KEY", "").strip()
    if llave:
        h["x-cg-demo-api-key"] = llave
    return h


def construir_url():
    return BASE + ENDPOINT + "?" + urllib.parse.urlencode(PARAMS)


def mostrar_peticion(url):
    """Imprime exactamente qué se está enviando (requisito del ejercicio)."""
    print("=" * 78)
    print("PETICIÓN HTTP")
    print("=" * 78)
    print("Método   : GET")
    print("Endpoint : " + BASE + ENDPOINT)
    print("Timeout  : {} s   |   Reintentos máximos: {}".format(TIEMPO_ESPERA, MAX_INTENTOS))
    print("\nParámetros (query string):")
    ancho = max(len(k) for k in PARAMS)
    for clave, valor in PARAMS.items():
        print("  {:<{w}} = {}".format(clave, valor, w=ancho))
    print("\nEncabezados:")
    for clave, valor in encabezados().items():
        if "api-key" in clave:                 # nunca imprimas la llave completa
            valor = valor[:6] + "..." + valor[-4:] if len(valor) > 12 else "***"
        print("  {}: {}".format(clave, valor))
    print("\nURL final:\n  " + url)
    print("=" * 78 + "\n")


def espera_sugerida(exc, intento):
    """Respeta Retry-After si el servidor lo manda; si no, backoff exponencial."""
    retry_after = None
    if isinstance(exc, urllib.error.HTTPError):
        retry_after = exc.headers.get("Retry-After")
    if retry_after:
        try:
            return max(1.0, float(retry_after))
        except ValueError:
            pass
    return ESPERA_BASE * (2 ** (intento - 1))


def descargar():
    """GET con manejo de errores de red, límites de uso (429) y fallas 5xx."""
    url = construir_url()
    mostrar_peticion(url)

    peticion = urllib.request.Request(url, headers=encabezados(), method="GET")
    contexto = ssl.create_default_context()

    for intento in range(1, MAX_INTENTOS + 1):
        print("[intento {}/{}] GET {}".format(intento, MAX_INTENTOS, ENDPOINT))
        try:
            with urllib.request.urlopen(peticion, timeout=TIEMPO_ESPERA,
                                        context=contexto) as resp:
                crudo = resp.read().decode("utf-8")
                restantes = resp.headers.get("x-ratelimit-remaining")
                print("  -> HTTP {} ({} bytes){}".format(
                    resp.status, len(crudo),
                    "  | llamadas restantes: " + restantes if restantes else ""))
            datos = json.loads(crudo)
            if not isinstance(datos, list) or not datos:
                raise ValueError("La API respondió sin datos utilizables.")
            return datos

        except urllib.error.HTTPError as e:
            cuerpo = ""
            try:
                cuerpo = e.read().decode("utf-8", "replace")[:200]
            except Exception:
                pass

            if e.code == 429:                      # LÍMITE DE USO EXCEDIDO
                pausa = espera_sugerida(e, intento)
                print("  -> HTTP 429: límite de uso de la API excedido. "
                      "Esperando {:.0f} s...".format(pausa))
            elif 500 <= e.code < 600:              # falla temporal del servidor
                pausa = espera_sugerida(e, intento)
                print("  -> HTTP {}: error del servidor. Reintentando en "
                      "{:.0f} s...".format(e.code, pausa))
            else:                                  # 400/401/403/404: no se arregla reintentando
                print("  -> HTTP {}: {}".format(e.code, e.reason))
                if cuerpo:
                    print("     respuesta: " + cuerpo)
                raise SystemExit("Petición inválida o rechazada; revisa los parámetros "
                                 "o la llave de API. No tiene caso reintentar.")

        except urllib.error.URLError as e:         # DNS, sin internet, TLS, proxy
            pausa = espera_sugerida(e, intento)
            print("  -> Error de red: {}. Reintentando en {:.0f} s...".format(e.reason, pausa))

        except TimeoutError:                       # el servidor no respondió a tiempo
            pausa = espera_sugerida(None, intento)
            print("  -> Timeout tras {} s. Reintentando en {:.0f} s...".format(
                TIEMPO_ESPERA, pausa))

        except json.JSONDecodeError as e:          # respuesta corrupta / portal cautivo
            pausa = espera_sugerida(None, intento)
            print("  -> La respuesta no es JSON válido ({}). Reintentando en "
                  "{:.0f} s...".format(e.msg, pausa))

        if intento == MAX_INTENTOS:
            raise SystemExit(
                "\nNo se pudo obtener la respuesta tras {} intentos.\n"
                "Sugerencias: revisa tu conexión, espera un minuto (el acceso público\n"
                "de CoinGecko permite pocas llamadas por minuto) o configura\n"
                "COINGECKO_API_KEY con una llave Demo gratuita.".format(MAX_INTENTOS))
        time.sleep(pausa)


# ================================================================ 2. DATAFRAME
def a_dataframe(datos):
    df = pd.json_normalize(datos, sep="_")     # aplana roi.times, roi.currency, etc.

    # json_normalize deja una columna "roi" vacía cuando el campo viene nulo en
    # algunas monedas; ya está desglosado en roi_times / roi_currency / roi_percentage.
    if "roi" in df.columns and "roi_times" in df.columns:
        df = df.drop(columns="roi")

    for col in ("last_updated", "ath_date", "atl_date"):   # ISO 8601 -> datetime UTC
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], format="ISO8601", utc=True)

    df.insert(0, "rango", np.arange(1, len(df) + 1))   # 1..20 por cap. de mercado
    df["market_cap_miles_millones"] = (df["market_cap"] / 1e9).round(2)
    return df


# ================================================================ 3. GRÁFICA
# Paleta y cromo consistentes con las gráficas del EJERCICIO3.
SURFACE, INK, INK_2, INK_MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, AZUL = "#e1e0d9", "#c3c2b7", "#2a78d6"

plt.rcParams.update({
    "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"],
    "text.parse_math": False,          # "$" es moneda, no matemáticas
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "text.color": INK,
    "axes.edgecolor": AXIS,
    "axes.labelcolor": INK_2,
    "xtick.color": INK_MUTED,
    "ytick.color": INK_MUTED,
    "axes.labelsize": 11.5,
    "xtick.labelsize": 10.5,
    "ytick.labelsize": 10.5,
    "axes.grid": False,
    "figure.dpi": 110,
})


def fmt_mm(v):
    """Miles de millones de USD, con la precisión justa para el tamaño."""
    if v >= 100:
        return format(v, ",.0f")
    if v >= 10:
        return format(v, ",.1f")
    return format(v, ",.2f")


def barra_redondeada(ax, y, largo, alto, color, rx, ry):
    """Barra horizontal con extremo de dato redondeado y base anclada al cero."""
    rx, ry = min(rx, abs(largo) / 2), min(ry, alto / 2)
    x0, x1, y0, y1 = 0.0, largo, y - alto / 2, y + alto / 2
    verts = [(x0, y0), (x1 - rx, y0), (x1, y0), (x1, y0 + ry),
             (x1, y1 - ry), (x1, y1), (x1 - rx, y1), (x0, y1), (x0, y0)]
    codes = [Path.MOVETO, Path.LINETO, Path.CURVE3, Path.CURVE3,
             Path.LINETO, Path.CURVE3, Path.CURVE3, Path.LINETO, Path.CLOSEPOLY]
    ax.add_patch(PathPatch(Path(verts, codes), facecolor=color,
                           edgecolor=SURFACE, linewidth=1.6, zorder=3))


def graficar(df, ruta):
    d = df.sort_values("market_cap", ascending=False)
    valores = (d["market_cap"] / 1e9).to_numpy(dtype=float)
    etiquetas = [n + "  " + s.upper() for n, s in zip(d["name"], d["symbol"])]

    y = np.arange(len(d))[::-1]        # la más grande hasta arriba
    fig, ax = plt.subplots(figsize=(12.4, 8.4))
    fig.subplots_adjust(left=0.235, right=0.965, top=0.855, bottom=0.085)

    ax.set_ylim(-0.75, len(d) - 0.25)
    ax.set_xlim(0, valores.max() * 1.12)   # aire para la etiqueta directa
    fig.canvas.draw()                      # fija transData antes de medir el radio

    inv = ax.transData.inverted()
    (px0, py0), (px1, py1) = inv.transform((0, 0)), inv.transform((4.0, 4.0))
    rx, ry = abs(px1 - px0), abs(py1 - py0)   # radio de 4 px en unidades de dato

    # Serie única: un solo tono. El valor va como etiqueta directa en cada barra,
    # porque el rango es tan amplio que las últimas barras miden pocos píxeles.
    for yi, v in zip(y, valores):
        barra_redondeada(ax, yi, v, 0.62, AZUL, rx, ry)
        ax.annotate(fmt_mm(v), (v, yi), textcoords="offset points", xytext=(9, 0),
                    va="center", ha="left", fontsize=10.5, fontweight="600", color=INK)

    ax.set_yticks(y)
    ax.set_yticklabels(etiquetas, fontsize=10.5, color=INK_2)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: format(v, ",.0f")))
    ax.set_xlabel("Capitalización de mercado (miles de millones de USD)")

    # Cromo recesivo: sin marco, rejilla hairline solo sobre el eje de valor.
    for lado in ("top", "right", "bottom"):
        ax.spines[lado].set_visible(False)
    ax.spines["left"].set_color(AXIS)
    ax.spines["left"].set_linewidth(1.0)
    ax.tick_params(length=0, pad=7)
    ax.set_axisbelow(True)
    ax.xaxis.grid(True, color=GRID, linewidth=1.0)

    total = valores.sum()
    dominio = valores[0] / total * 100
    fig.text(0.035, 0.968, "Las 20 criptomonedas más grandes por capitalización de mercado",
             fontsize=21, fontweight="600", color=INK, va="top")
    fig.text(0.035, 0.922,
             "{} concentra el {:.0f}% de los {:,.0f} mil millones de USD del top 20"
             .format(d["name"].iloc[0], dominio, total),
             fontsize=12, color=INK_2, va="top")
    fig.text(0.035, 0.022,
             "Fuente: CoinGecko /coins/markets · datos al " +
             d["last_updated"].max().strftime("%Y-%m-%d %H:%M UTC"),
             fontsize=9.5, color=INK_MUTED, va="bottom")

    fig.savefig(ruta, dpi=200)
    plt.close(fig)


# ================================================================ MAIN
def main():
    datos = descargar()
    df = a_dataframe(datos)

    print("\n" + "=" * 78)
    print("DATAFRAME  ({} filas x {} columnas)".format(*df.shape))
    print("=" * 78)
    vista = df[["rango", "name", "symbol", "current_price",
                "market_cap_miles_millones", "price_change_percentage_24h"]].copy()
    vista.columns = ["#", "Nombre", "Simbolo", "Precio USD",
                     "Cap. (miles de millones USD)", "Var. 24h %"]
    print(vista.to_string(index=False,
                          formatters={"Precio USD": lambda v: format(v, ",.4f"),
                                      "Var. 24h %": lambda v: format(v, "+.2f")}))

    df.to_csv(SALIDA_CSV, index=False, encoding="utf-8-sig")
    print("\nCSV guardado    : {}  ({} columnas)".format(SALIDA_CSV, df.shape[1]))

    graficar(df, SALIDA_PNG)
    print("Gráfica guardada: " + SALIDA_PNG)


if __name__ == "__main__":
    sys.exit(main())
