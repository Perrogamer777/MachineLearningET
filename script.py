from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Ruta de los archivos del modelo y las columnas
model_path = 'checkpoints/tree_model.pkl'
columns_path = 'checkpoints/columns.pkl'

# Cargar el modelo
with open(model_path, 'rb') as model_file:
    model = pickle.load(model_file)

# Cargar los nombres de las columnas
with open(columns_path, 'rb') as columns_file:
    columns = pickle.load(columns_file)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Obtener los datos del formulario
        map_value = request.form['map']
        team_value = request.form['team']
        round_kills = int(request.form['round_kills'])
        survived = int(request.form['survived'])

        # Crear el DataFrame con los datos de entrada
        input_data = {
            'RoundKills': [round_kills],
            'Survived': [survived],
            f'Map_{map_value}': [1],
            f'Team_{team_value}': [1]
        }

        # Completar las columnas faltantes con ceros
        for col in columns:
            if col not in input_data:
                input_data[col] = [0]

        # Convertir a DataFrame y ordenar las columnas
        input_df = pd.DataFrame(input_data)
        input_df = input_df[columns]

        # Realizar la predicción
        prediction = model.predict(input_df)[0]

        return render_template('resultado.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
