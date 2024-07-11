import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

pickle_path = 'checkpoints/dataframe.pkl'
with open(pickle_path, 'rb') as file:
    df = pickle.load(file)

try:
    # Inspeccionar los nombres de las columnas
    print("Columnas disponibles en el DataFrame:")
    print(df.columns)

    # Seleccionar las columnas relevantes incluyendo 'RoundKills' y 'Survived'
    datosUtilizar = df[['Map', 'Team', 'RoundWinner', 'RoundKills', 'Survived']].copy()

    # Convertir las columnas 'RoundWinner' y 'Survived' en valores binarios (0 y 1)
    datosUtilizar['RoundWinner'] = datosUtilizar['RoundWinner'].astype(int)
    datosUtilizar['Survived'] = datosUtilizar['Survived'].astype(int)

    # Convertir columnas categóricas en variables dummy
    data_dummies = pd.get_dummies(datosUtilizar, columns=['Map', 'Team'], drop_first=True)

    # Verificar el balance de clases
    class_counts = data_dummies['RoundWinner'].value_counts()
    print("Distribución de clases:\n", class_counts)

    # Separar características y variable objetivo
    X = data_dummies.drop('RoundWinner', axis=1)
    y = data_dummies['RoundWinner']

    # Dividir los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Crear y entrenar el modelo de árbol de decisiones
    tree_model = DecisionTreeClassifier(random_state=42)
    tree_model.fit(X_train, y_train)

    # Realizar predicciones
    y_pred = tree_model.predict(X_test)

    # Evaluación del modelo
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)

    print("Precisión del modelo de Árbol de Decisiones:", accuracy)
    print("Matriz de confusión:\n", conf_matrix)
    print("Reporte de clasificación:\n", class_report)

    # Guardar el modelo y los nombres de las columnas en archivos pickle
    with open('checkpoints/tree_model.pkl', 'wb') as model_file:
        pickle.dump(tree_model, model_file)
        
    with open('checkpoints/columns.pkl', 'wb') as columns_file:
        pickle.dump(X_train.columns.tolist(), columns_file)

    print("todo salió de maravilla perro B)")

except pd.errors.ParserError as e:
    print("Error al analizar el archivo CSV:", e)
except KeyError as e:
    print("Error de clave:", e)
except Exception as e:
    print("Ocurrió un error:", e)
