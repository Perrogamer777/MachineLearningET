from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle

# Limpieza de datos numéricos 
def limpiar(column):
    return pd.to_numeric(column.str.replace('.', '', regex=True), errors='coerce')

# Seleccionar las columnas relevantes
datosUtilizar = df_filtrado[['Map', 'Team', 'RoundWinner']].copy()

# Convertir la columna 'RoundWinner' en valores binarios 
datosUtilizar['RoundWinner'] = datosUtilizar['RoundWinner'].astype(int)

# Convertir columnas categóricas en variables dummy
data_dummies = pd.get_dummies(datosUtilizar, columns=['Map', 'Team'], drop_first=True)

# Separar características y variable objetivo
X = data_dummies.drop('RoundWinner', axis=1)
y = data_dummies['RoundWinner']

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Ajustar el modelo de regresión logística
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# predicción
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

#evaluar modelo
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f'Precisión: {accuracy}')
print(f'Matriz de confusión: \n{conf_matrix}')
print(f'Reporte de clasificación: \n{class_report}')


# Guardar el modelo y el scaler en archivos pickle
with open('checkpoints/logistic_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)


print("Modelo y scaler guardados en archivos pickle con éxito.")
