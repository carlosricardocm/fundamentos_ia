Tengo un conjunto de datos de abandono de clientes en churn_clientes_features.csv. La variable
objetivo es abandono. Construye un pipeline de machine learning completo:
1. Divide en entrenamiento y prueba 80/20 de forma estratificada.
2. Imputa los valores faltantes y codifica las variables categóricas.
3. Escala las variables numéricas.
4. Entrena y compara regresión logística, random forest y gradient boosting.
5. Reporta exactitud, precisión, recall y F1 con validación cruzada.
6. Ajusta los hiperparámetros del mejor modelo con búsqueda en malla.
7. Muestra la importancia de las variables.
8. Guarda el mejor modelo como modelo_churn.pkl junto con la lista de variables usadas.
Dime qué modelo funciona mejor y por qué, y advierteme si las métricas son sospechosamente alta