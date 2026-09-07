Tengo un conjunto de datos de clientes muy sucio en clientes_sucios.csv. Límpialo por completo:
1. Estandariza nombre_cliente a formato Título.
2. Valida el formato de email y marca los inválidos en una columna nueva.
3. Estandariza telefono al formato (555) 123-4567.
4. Convierte todos los valores de fecha_registro al formato AAAA-MM-DD.
5. Limpia ingreso_anual: quita el $ y las comas y conviértelo a número decimal.
6. Estandariza pais al nombre completo del país.
7. Corrige edad: elimina edades imposibles (menores que 0 o mayores que 100).
8. Estandariza activo a valores booleanos verdadero/falso.
9. Elimina las filas duplicadas.
Muéstrame una comparación antes/después con el número de filas y columnas, y guarda el resultado como
clientes_limpios.csv.