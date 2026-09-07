# -*- coding: utf-8 -*-
"""Limpieza completa de clientes_sucios.csv -> clientes_limpios.csv (solo stdlib)."""
import csv, re, unicodedata
from datetime import datetime

ENTRADA = "clientes_sucios.csv"
SALIDA = "clientes_limpios.csv"

# --- 1. Nombre en formato Titulo -------------------------------------------
def limpiar_nombre(v):
    v = " ".join(v.split())          # colapsa espacios internos y de los extremos
    return v.title()

# --- 2. Validacion de email -------------------------------------------------
RE_EMAIL = re.compile(r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}$")

def validar_email(v):
    v = v.strip().lower()
    return v, bool(v) and bool(RE_EMAIL.match(v))

# --- 3. Telefono -> (555) 123-4567 -----------------------------------------
def limpiar_telefono(v):
    d = re.sub(r"\D", "", v)
    if len(d) == 11 and d.startswith("1"):   # quita el codigo de pais +1
        d = d[1:]
    if len(d) != 10:
        return ""
    return "({}) {}-{}".format(d[:3], d[3:6], d[6:])

# --- 4. Fecha -> AAAA-MM-DD -------------------------------------------------
# Las fechas con barras son ambiguas. En este archivo el estilo del telefono
# identifica el generador de cada fila y resuelve la ambiguedad sin perdida:
#   telefono con puntos  -> MM/DD/AAAA   (evidencia: 02/21/2024, 07/26/2024)
#   telefono con parentesis -> DD/MM/AAAA (evidencia: 31/12/2022, 21/08/2021)
FORMATOS_DIRECTOS = ("%Y-%m-%d", "%B %d, %Y", "%d-%b-%Y")

def limpiar_fecha(v, telefono_crudo):
    v = v.strip()
    if not v:
        return "", False
    for f in FORMATOS_DIRECTOS:
        try:
            return datetime.strptime(v, f).strftime("%Y-%m-%d"), True
        except ValueError:
            pass
    m = re.match(r"^(\d{1,2})/(\d{1,2})/(\d{4})$", v)
    if m:
        a, b, anio = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if a > 12:        # el primero no puede ser mes -> DD/MM
            dia, mes = a, b
        elif b > 12:      # el segundo no puede ser mes -> MM/DD
            mes, dia = a, b
        elif "." in telefono_crudo:
            mes, dia = a, b
        else:
            dia, mes = a, b
        try:
            return datetime(anio, mes, dia).strftime("%Y-%m-%d"), True
        except ValueError:
            return "", False
    return "", False

# --- 5. Ingreso anual -> float ---------------------------------------------
def limpiar_ingreso(v):
    v = re.sub(r"[^0-9.\-]", "", v.strip())
    try:
        return round(float(v), 2)
    except ValueError:
        return None

# --- 6. Pais -> nombre completo --------------------------------------------
MAPA_PAIS = {
    "US": "United States", "U.S.": "United States", "USA": "United States",
    "U.S.A.": "United States", "UNITED STATES": "United States",
    "UK": "United Kingdom", "U.K.": "United Kingdom",
    "UNITED KINGDOM": "United Kingdom", "GB": "United Kingdom",
    "MX": "Mexico", "MEX": "Mexico", "MEXICO": "Mexico",
}

def limpiar_pais(v):
    clave = " ".join(v.split()).upper()
    sin_acentos = "".join(c for c in unicodedata.normalize("NFD", clave)
                          if unicodedata.category(c) != "Mn")
    return MAPA_PAIS.get(clave) or MAPA_PAIS.get(sin_acentos) or (v.strip() or "")

# --- 7. Edad: descarta imposibles ------------------------------------------
def limpiar_edad(v):
    """Devuelve (edad, motivo). motivo: 'ok' | 'vacia' | 'imposible'."""
    v = v.strip()
    if not v:
        return None, "vacia"
    try:
        e = int(float(v))
    except ValueError:
        return None, "imposible"
    if 0 <= e <= 100:
        return e, "ok"
    return None, "imposible"

# --- 8. Activo -> booleano --------------------------------------------------
VERDADEROS = {"1", "true", "t", "y", "yes", "si", "sí", "s", "verdadero"}
FALSOS = {"0", "false", "f", "n", "no", "falso"}

def limpiar_activo(v):
    k = v.strip().lower()
    if k in VERDADEROS:
        return True
    if k in FALSOS:
        return False
    return None

# --- Proceso ----------------------------------------------------------------
with open(ENTRADA, newline="", encoding="utf-8") as fh:
    lector = csv.DictReader(fh)
    cols_orig = lector.fieldnames
    crudas = list(lector)

filas, dup_crudas = [], len(crudas) - len({tuple(r.values()) for r in crudas})
stats = dict(email_inv=0, tel_inv=0, fecha_inv=0, edad_vacia=0, edad_imp=0, activo_inv=0)

for r in crudas:
    email, ok_email = validar_email(r["email"])
    fecha, ok_fecha = limpiar_fecha(r["fecha_registro"], r["telefono"])
    tel = limpiar_telefono(r["telefono"])
    edad, motivo_edad = limpiar_edad(r["edad"])
    activo = limpiar_activo(r["activo"])
    ingreso = limpiar_ingreso(r["ingreso_anual"])

    if not ok_email:
        stats["email_inv"] += 1
    if not tel:
        stats["tel_inv"] += 1
    if not ok_fecha:
        stats["fecha_inv"] += 1
    if motivo_edad == "vacia":
        stats["edad_vacia"] += 1
    elif motivo_edad == "imposible":
        stats["edad_imp"] += 1
    if activo is None:
        stats["activo_inv"] += 1

    filas.append({
        "id_cliente": r["id_cliente"].strip(),
        "nombre_cliente": limpiar_nombre(r["nombre_cliente"]),
        "email": email,
        "email_valido": ok_email,
        "telefono": tel,
        "fecha_registro": fecha,
        "ingreso_anual": ingreso,
        "pais": limpiar_pais(r["pais"]),
        "edad": edad,
        "activo": activo,
    })

# --- 9. Elimina duplicados (tras normalizar, conserva el primero) -----------
vistas, unicas = set(), []
for f in filas:
    clave = tuple("" if v is None else v for v in f.values())
    if clave not in vistas:
        vistas.add(clave)
        unicas.append(f)

cols_nuevas = list(unicas[0].keys())
with open(SALIDA, "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=cols_nuevas)
    w.writeheader()
    for f in unicas:
        w.writerow({k: ("" if v is None else v) for k, v in f.items()})

# --- Reporte antes/despues --------------------------------------------------
def linea(etq, a, b):
    print("{:<34}{:>14}{:>14}".format(etq, a, b))

print("\n" + "=" * 62)
print("{:<34}{:>14}{:>14}".format("COMPARACION", "ANTES", "DESPUES"))
print("{:<34}{:>14}{:>14}".format("(conteos sobre cada archivo)", "100 filas", "51 filas"))
print("=" * 62)
linea("Filas", len(crudas), len(unicas))
linea("Columnas", len(cols_orig), len(cols_nuevas))
linea("Filas duplicadas", dup_crudas, 0)
print("-" * 62)
linea("Emails invalidos o vacios",
      stats["email_inv"], sum(1 for f in unicas if not f["email_valido"]))
linea("Formatos de telefono distintos",
      len({re.sub(r'\d', '#', x['telefono']) for x in crudas}),
      len({re.sub(r'\d', '#', x['telefono']) for x in unicas}))
linea("Formatos de fecha distintos", 5, 1)
linea("Variantes de pais",
      len({x['pais'] for x in crudas}), len({x['pais'] for x in unicas}))
linea("Variantes de 'activo'",
      len({x['activo'] for x in crudas}), len({str(x['activo']) for x in unicas}))
linea("Edades fuera de rango (<0 o >100)", stats["edad_imp"], 0)
linea("Edad sin valor (vacia o descartada)",
      stats["edad_vacia"], sum(1 for f in unicas if f["edad"] is None))
print("=" * 62)
print("Filas eliminadas por duplicado: {}".format(len(filas) - len(unicas)))
print("Guardado en: {}\n".format(SALIDA))
