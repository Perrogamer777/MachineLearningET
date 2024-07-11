import pandas as pd
import numpy as np

data_path = 'dataset/Anexo_ET_demo_round_traces.csv'
df = pd.read_csv(data_path, sep=';')

## Se selecciona las partidas que tengan más de 30 rondas y posteriormente se eliminan.
rounds_mayores_a_30 = df.loc[df['RoundId'] > 30]
df = df.drop(rounds_mayores_a_30.index)
round_por_match = df.groupby('MatchId')['RoundId'].max().reset_index().sort_values(by='RoundId')
max_rounds = round_por_match['RoundId']

## Se elimina la columna AbnormalMatch por su nulo aporte
df = df.drop('AbnormalMatch', axis=1)
##Se convierte la columna RoundWinner en valores binarios
df['RoundWinner'] = df['RoundWinner'].astype(bool)

## Se modifica la columna 'Team' en función de la ronda y el equipo interno para asignar los valores "Terrorist" y "CounterTerrorist" según el momento del juego
for partida in df['MatchId'].unique():
    team1 = df[(df['RoundId'] == 1) & (df['InternalTeamId'] == 1) & (df['MatchId'] == partida)]['Team'].unique()[0]
    team2 = df[(df['RoundId'] == 1) & (df['InternalTeamId'] == 2) & (df['MatchId'] == partida)]['Team'].unique()[0]
    rondas = df[(df['MatchId'] == partida)]['RoundId'].unique()
    for ronda in rondas:
        if ronda < 16:
            df.loc[(df['RoundId'] == ronda) & (df['InternalTeamId'] == 1) & (df['MatchId'] == partida), 'Team'] = 'Terrorist'
            df.loc[(df['RoundId'] == ronda) & (df['InternalTeamId'] == 2) & (df['MatchId'] == partida), 'Team'] = 'CounterTerrorist'
        else:
            df.loc[(df['RoundId'] == ronda) & (df['InternalTeamId'] == 1) & (df['MatchId'] == partida), 'Team'] = 'CounterTerrorist'
            df.loc[(df['RoundId'] == ronda) & (df['InternalTeamId'] == 2) & (df['MatchId'] == partida), 'Team'] = 'Terrorist'


# Indicamos qué equipo ganó la partida y el recuento de partidas ganadas por cada equipo y de partidas empatadas.
partidas_ganadas_equipo1 = 0
partidas_ganadas_equipo2 = 0
partidas_empatadas = 0


for partida in df['MatchId'].unique():
    rondas = df[(df['MatchId'] == partida)]['RoundId'].unique()
    equipo1_rondas_ganadas = len(df[(df['MatchId'] == partida) & (df['InternalTeamId'] == 1) & (df['RoundWinner'] == True)])
    equipo2_rondas_ganadas = len(df[(df['MatchId'] == partida) & (df['InternalTeamId'] == 2) & (df['RoundWinner'] == True)])
    if equipo1_rondas_ganadas > equipo2_rondas_ganadas:
        df.loc[(df['MatchId'] == partida) & (df['InternalTeamId'] == 1), 'MatchWinner'] = True
        df.loc[(df['MatchId'] == partida) & (df['InternalTeamId'] == 2), 'MatchWinner'] = False
        partidas_ganadas_equipo1 += 1
    elif equipo1_rondas_ganadas < equipo2_rondas_ganadas:
        df.loc[(df['MatchId'] == partida) & (df['InternalTeamId'] == 1), 'MatchWinner'] = False
        df.loc[(df['MatchId'] == partida) & (df['InternalTeamId'] == 2), 'MatchWinner'] = True
        partidas_ganadas_equipo2 += 1
    else:
        # En caso de empate en rondas ganadas, dejar datos como nan
        df.loc[df['MatchId'] == partida, 'MatchWinner'] = np.nan
        partidas_empatadas += 1

print("Partidas ganadas empezando como Terrorist:", partidas_ganadas_equipo1)
print("Partidas ganadas empezando como CounterTerrorist:", partidas_ganadas_equipo2)
print("Partidas empatadas:", partidas_empatadas)


### Obtenemos los datos de las partidas donde se gana comenzando como terrorista y counter terrorist
mapas = df['Map'].unique()

for mapa in mapas:
    partidas_ganadas_equipo1 = 0
    partidas_ganadas_equipo2 = 0
    partidas_empatadas = 0

    # Filtrar el df por el mapa
    df_mapa = df[df['Map'] == mapa]

    for partida in df_mapa['MatchId'].unique():
        rondas = df_mapa[(df_mapa['MatchId'] == partida)]['RoundId'].unique()
        equipo1_rondas_ganadas = len(df_mapa[(df_mapa['MatchId'] == partida) & (df_mapa['InternalTeamId'] == 1) & (df_mapa['RoundWinner'] == True)])
        equipo2_rondas_ganadas = len(df_mapa[(df_mapa['MatchId'] == partida) & (df_mapa['InternalTeamId'] == 2) & (df_mapa['RoundWinner'] == True)])

        if equipo1_rondas_ganadas > equipo2_rondas_ganadas:
            partidas_ganadas_equipo1 += 1
        elif equipo1_rondas_ganadas < equipo2_rondas_ganadas:
            partidas_ganadas_equipo2 += 1
        else:
            partidas_empatadas += 1

    print("Mapa:", mapa)
    print("Partidas ganadas empezando como Terroristas:", partidas_ganadas_equipo1)
    print("Partidas ganadas empezando como CounterTerrorist:", partidas_ganadas_equipo2)
    print("Partidas empatadas:", partidas_empatadas)
    print()

### Se obtienen los datos de rondas ganadas por equipo y mapa
mapas = df['Map'].unique()
total_rondas_ganadas_terrorist = []
total_rondas_ganadas_counterterrorist = []

# Calcular el total de rondas ganadas por equipo por mapa
for mapa in mapas:
    df_mapa = df[df['Map'] == mapa]
    total_rondas_ganadas_terrorist.append(df_mapa[df_mapa['Team'] == 'Terrorist']['RoundWinner'].sum())
    total_rondas_ganadas_counterterrorist.append(df_mapa[df_mapa['Team'] == 'CounterTerrorist']['RoundWinner'].sum())

x = np.arange(len(mapas))
width = 0.35

pickle_path = 'checkpoints/dataframe.pkl'
df.to_pickle(pickle_path)

