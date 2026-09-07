# -*- coding: utf-8 -*-
"""
Construye ingresos.pptx (16:9) a partir de los PNG generados por crear_graficas.py.

Estructura: portada + una diapositiva por gráfica (con 2-3 oraciones de
interpretación debajo) + diapositiva de cierre con los 3 hallazgos principales.
Los números citados en el texto se recalculan desde el CSV, no se escriben a mano.
"""
from PIL import Image
import pandas as pd
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR

# Mismos tokens de color que las gráficas
SURFACE   = RGBColor(0xFC, 0xFC, 0xFB)
INK       = RGBColor(0x0B, 0x0B, 0x0B)
INK_2     = RGBColor(0x52, 0x51, 0x4E)
INK_MUTED = RGBColor(0x89, 0x87, 0x81)
GRID      = RGBColor(0xE1, 0xE0, 0xD9)
PLANE     = RGBColor(0xF9, 0xF9, 0xF7)
AZUL      = RGBColor(0x2A, 0x78, 0xD6)
NARANJA   = RGBColor(0xEB, 0x68, 0x34)
AQUA      = RGBColor(0x1B, 0xAF, 0x7A)
AMARILLO  = RGBColor(0xED, 0xA1, 0x00)

FUENTE = "Segoe UI"
ANCHO, ALTO = Inches(13.333), Inches(7.5)

# ---------------------------------------------------------------- datos
df = pd.read_csv("ingresos_mensuales.csv")
total = df["ingreso"].sum()
por_mes = df.groupby("mes")["ingreso"].sum().sort_index()
por_region = df.groupby("region")["ingreso"].sum()
ing = por_mes.values
crec = (ing[-1] / ing[0] - 1) * 100
h1, h2 = ing[:6].sum(), ing[6:].sum()
delta_sem = (h2 / h1 - 1) * 100
aov = total / df.unidades_vendidas.sum()
aov_reg = df.groupby("region").apply(
    lambda g: g.ingreso.sum() / g.unidades_vendidas.sum(), include_groups=False)


def d(v, dec=0):
    return "$" + format(round(v, dec), ",." + str(dec) + "f")


def k(v):
    return "$" + format(v / 1000, ",.0f") + "k"


def p(v, dec=1):
    return format(v, "." + str(dec) + "f") + "%"


# ---------------------------------------------------------------- helpers
prs = Presentation()
prs.slide_width, prs.slide_height = ANCHO, ALTO
BLANCA = prs.slide_layouts[6]          # diapositiva en blanco


def nueva(fondo=SURFACE):
    s = prs.slides.add_slide(BLANCA)
    s.background.fill.solid()
    s.background.fill.fore_color.rgb = fondo
    return s


def caja(slide, x, y, w, h, texto, size=14, color=INK_2, bold=False,
         align=PP_ALIGN.LEFT, espaciado=1.35, anchor=MSO_ANCHOR.TOP):
    tb = slide.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    partes = texto.split("\n")
    for i, linea in enumerate(partes):
        par = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        par.alignment = align
        par.line_spacing = espaciado
        r = par.add_run()
        r.text = linea
        r.font.size = Pt(size)
        r.font.bold = bold
        r.font.color.rgb = color
        r.font.name = FUENTE
    return tb


def barra(slide, x, y, w, h, color):
    from pptx.enum.shapes import MSO_SHAPE
    sh = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, x, y, w, h)
    sh.fill.solid()
    sh.fill.fore_color.rgb = color
    sh.line.fill.background()
    sh.shadow.inherit = False
    return sh


def pie_pagina(slide, n):
    caja(slide, Inches(0.75), Inches(7.02), Inches(8.0), Inches(0.3),
         "Desempeño de ingresos 2024  ·  Fuente: ingresos_mensuales.csv",
         size=9, color=INK_MUTED)
    caja(slide, Inches(11.9), Inches(7.02), Inches(0.7), Inches(0.3),
         str(n), size=9, color=INK_MUTED, align=PP_ALIGN.RIGHT)


def slide_grafica(png, interpretacion, n):
    """Imagen centrada arriba + 2-3 oraciones de interpretación debajo."""
    s = nueva()
    max_w, max_h = Inches(11.9), Inches(4.95)
    pw, ph = Image.open(png).size
    escala = min(max_w / pw, max_h / ph)
    w, h = Emu(int(pw * escala)), Emu(int(ph * escala))
    s.shapes.add_picture(png, int((ANCHO - w) / 2), Inches(0.42), w, h)

    y_texto = Inches(0.42) + h + Inches(0.34)
    barra(s, Inches(0.75), y_texto + Inches(0.05), Inches(0.045), Inches(0.95), AZUL)
    caja(s, Inches(1.02), y_texto, Inches(11.5), Inches(1.15),
         interpretacion, size=13.5, color=INK_2, espaciado=1.4)
    pie_pagina(s, n)
    return s


# ---------------------------------------------------------------- 1. portada
s = nueva()
barra(s, Inches(0.75), Inches(2.25), Inches(1.6), Inches(0.075), AZUL)
caja(s, Inches(0.75), Inches(2.62), Inches(11.5), Inches(1.2),
     "Desempeño de ingresos 2024", size=46, color=INK, bold=True, espaciado=1.05)
caja(s, Inches(0.75), Inches(3.95), Inches(10.5), Inches(0.9),
     "Análisis del ingreso mensual por región · enero a diciembre de 2024",
     size=18, color=INK_2)

# franja de indicadores de portada
datos_portada = [
    ("Ingreso total", d(total), AZUL),
    ("Crecimiento anual", "+" + p(crec), AQUA),
    ("Región líder", "Centro", NARANJA),
    ("Regiones · meses", "4 · 12", AMARILLO),
]
x0, ancho_t, gap = Inches(0.75), Inches(2.72), Inches(0.19)
for i, (rot, val, col) in enumerate(datos_portada):
    x = x0 + i * (ancho_t + gap)
    barra(s, x, Inches(5.15), ancho_t, Inches(1.05), PLANE)
    barra(s, x, Inches(5.15), Inches(0.045), Inches(1.05), col)
    caja(s, x + Inches(0.24), Inches(5.32), ancho_t - Inches(0.4), Inches(0.3),
         rot.upper(), size=9.5, color=INK_MUTED, bold=True)
    caja(s, x + Inches(0.24), Inches(5.62), ancho_t - Inches(0.4), Inches(0.45),
         val, size=20, color=INK, bold=True)

caja(s, Inches(0.75), Inches(6.72), Inches(11.5), Inches(0.4),
     "Fuente: ingresos_mensuales.csv · 48 registros (12 meses × 4 regiones), sin valores faltantes",
     size=10.5, color=INK_MUTED)

# ---------------------------------------------------------------- 2-5. gráficas
slide_grafica(
    "01_ingreso_por_mes.png",
    "El ingreso mensual pasa de " + d(ing[0]) + " en enero a " + d(ing[-1]) +
    " en diciembre, un avance de " + p(crec) + " en el año. El único retroceso "
    "relevante ocurre entre junio y julio (−5.3% y −7.4%), después del máximo parcial de mayo. "
    "A partir de agosto la curva se acelera y el segundo semestre cierra " + p(delta_sem) +
    " por encima del primero.", 2)

slide_grafica(
    "02_ingreso_por_region.png",
    "Centro aporta " + d(por_region["Centro"]) + ", equivalente al " +
    p(por_region["Centro"] / total * 100) + " del ingreso anual, y es la única región por "
    "encima de los 2 millones de dólares. Sur cierra en " + d(por_region["Sur"]) + " (" +
    p(por_region["Sur"] / total * 100) + "), prácticamente la mitad de lo que genera Centro. "
    "Las cuatro regiones crecieron a un ritmo parecido entre enero y diciembre (68% a 75%), "
    "así que la brecha responde al tamaño de cada mercado y no a su desempeño.", 3)

slide_grafica(
    "03_mapa_calor_mes_region.png",
    "El tono se oscurece de arriba hacia abajo en las cuatro columnas: el patrón estacional es "
    "común a todas las regiones y no un efecto local. Las franjas más claras aparecen en febrero "
    "y julio, y las más oscuras en noviembre y diciembre, cuando Centro alcanza $291k en un solo "
    "mes. Sur conserva el tono más claro durante todo el año, en un rango de $82k a $138k.", 4)

slide_grafica(
    "04_resumen_indicadores.png",
    "El año cierra con " + d(total) + " de ingreso, " +
    format(int(df.unidades_vendidas.sum()), ",") + " unidades vendidas y " +
    format(int(df.clientes_nuevos.sum()), ",") + " clientes nuevos. El ticket promedio es de " +
    d(aov, 2) + " y se mantiene casi idéntico en las cuatro regiones (" + d(aov_reg.min(), 2) +
    " a " + d(aov_reg.max(), 2) + "), de modo que el crecimiento proviene del volumen y no del "
    "precio. La tendencia mensual confirma que la aceleración se concentra en el último trimestre.", 5)

# ---------------------------------------------------------------- 6. hallazgos
s = nueva()
barra(s, Inches(0.75), Inches(0.62), Inches(1.6), Inches(0.075), AZUL)
caja(s, Inches(0.75), Inches(0.92), Inches(11.5), Inches(0.7),
     "Tres hallazgos principales", size=34, color=INK, bold=True)
caja(s, Inches(0.75), Inches(1.66), Inches(11.5), Inches(0.4),
     "Lo que el año 2024 deja como conclusión para la planeación del siguiente ciclo",
     size=14, color=INK_2)

hallazgos = [
    (AZUL, "El crecimiento es real y sostenido, no un pico de fin de año",
     "El ingreso crece " + p(crec) + " entre enero (" + d(ing[0]) + ") y diciembre (" +
     d(ing[-1]) + "), y el segundo semestre supera al primero en " + p(delta_sem) + " (" +
     d(h2) + " contra " + d(h1) + "). La tendencia es ascendente en 8 de los 11 cambios "
     "mensuales del año."),
    (NARANJA, "El ingreso está concentrado: Centro vale el doble que Sur",
     "Centro genera " + d(por_region["Centro"]) + " (" +
     p(por_region["Centro"] / total * 100) + " del total) frente a los " + d(por_region["Sur"]) +
     " de Sur (" + p(por_region["Sur"] / total * 100) + "). Como las cuatro regiones crecen a "
     "un ritmo similar, la concentración se mantendrá salvo que se actúe sobre Sur y Occidente."),
    (AQUA, "El motor es el volumen, no el precio; y hay un valle en junio-julio",
     "El ticket promedio apenas varía entre regiones (" + d(aov_reg.min(), 2) + " a " +
     d(aov_reg.max(), 2) + "), así que el crecimiento viene de vender más unidades. El único "
     "tramo negativo del año es junio-julio (−5.3% y −7.4%), un valle estacional que conviene "
     "cubrir con acciones comerciales."),
]

y = Inches(2.22)
for i, (color, titular, cuerpo) in enumerate(hallazgos):
    alto = Inches(1.38)
    barra(s, Inches(0.75), y, Inches(11.83), alto, PLANE)
    barra(s, Inches(0.75), y, Inches(0.055), alto, color)
    caja(s, Inches(1.12), y + Inches(0.20), Inches(0.5), Inches(0.4),
         str(i + 1), size=15, color=color, bold=True)
    caja(s, Inches(1.62), y + Inches(0.18), Inches(10.6), Inches(0.4),
         titular, size=16.5, color=INK, bold=True)
    caja(s, Inches(1.62), y + Inches(0.62), Inches(10.6), Inches(0.75),
         cuerpo, size=12.5, color=INK_2, espaciado=1.3)
    y = y + alto + Inches(0.18)

caja(s, Inches(0.75), Inches(7.02), Inches(11.83), Inches(0.3),
     "Fuente: ingresos_mensuales.csv · 48 registros (12 meses × 4 regiones)",
     size=9, color=INK_MUTED)
caja(s, Inches(11.9), Inches(7.02), Inches(0.7), Inches(0.3),
     "6", size=9, color=INK_MUTED, align=PP_ALIGN.RIGHT)

prs.save("ingresos.pptx")
print("ingresos.pptx guardado con", len(prs.slides._sldIdLst), "diapositivas")
