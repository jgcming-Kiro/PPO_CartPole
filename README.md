# PPO en CartPole-v1

Este entregable contiene un notebook reproducible para entrenar un agente PPO con Stable-Baselines3 en el entorno `CartPole-v1`.

Archivos principales:

- `PPO_CartPole_StableBaselines3.ipynb`: cuaderno con explicación, código, entrenamiento, evaluación y gráficas.
- `results/baseline_vs_ppo_summary.csv`: comparación entre baseline aleatorio y PPO.
- `results/ppo_learning_curve.csv`: retornos registrados durante el entrenamiento.
- `results/ppo_learning_curve.png`: curva de aprendizaje.
- `results/baseline_vs_ppo.png`: comparación visual de desempeño.
- `results/ppo_cartpole_model.zip`: modelo entrenado.

Resultado obtenido con `SEED = 42`:

- Baseline aleatorio: retorno promedio de 25.73 en 30 episodios.
- PPO entrenado: retorno promedio de 500.00 en 30 episodios.

Para repetir el experimento, abra el notebook y ejecute las celdas en orden.

Instalacion sugerida en una terminal, desde esta carpeta. Estas versiones son compatibles con Python 3.9:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m ipykernel install --user --name ppo-cartpole --display-name "Python PPO CartPole"
```

Luego abra `PPO_CartPole_StableBaselines3.ipynb` y seleccione el kernel `Python PPO CartPole`.

Tambien puede ejecutar el experimento completo sin abrir Jupyter:

```bash
python3 run_project.py
```

Ese comando vuelve a generar las tablas, el modelo y las graficas dentro de `results`.
