# Projeto Final IC - Massive MIMO com Reinforcement Learning

Este projeto treina e compara agentes de aprendizado por reforco para alocacao de potencia em um ambiente Massive MIMO. O notebook principal usa Stable-Baselines3 para treinar PPO, DQN e A2C em configuracoes padrao e com ajuste de hiperparametros via Optuna.

## Estrutura

- `run_vasco.ipynb`: notebook principal de treino, tuning, avaliacao e graficos.
- `network/`: ambiente `MassiveMIMOEnv` e utilitarios de simulacao.
- `check_environment.py`: validacao rapida das dependencias e do ambiente.
- `requirements.txt`: dependencias Python do projeto.

Os diretorios `results/`, `figures/` e `models/` sao gerados ao executar o notebook e nao sao versionados.

## Instalação

Use Python 3.11 ou superior.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

Se for usar GPU CUDA, instale o PyTorch adequado para sua versao de CUDA antes ou depois das dependencias, seguindo a documentacao oficial do PyTorch.

## Verificação

```bash
python check_environment.py
```

O comando deve imprimir as versoes das bibliotecas e finalizar com `Environment check: OK`.

## Execução

Abra `run_vasco.ipynb` no Jupyter/VS Code/Colab e execute as celulas em ordem.

No notebook:

- `FAST_MODE = True` roda um teste rapido.
- `FAST_MODE = False` roda a configuracao final, mais demorada.
- Cada execucao cria um `RUN_ID` e salva artefatos em:
  - `results/runs/<RUN_ID>/`
  - `figures/runs/<RUN_ID>/`
  - `models/runs/<RUN_ID>/`

Tambem sao criadas copias dos artefatos mais recentes em `results/`, `figures/` e `models/`.

## Resultados Gerados

O notebook gera tabelas como:

- `summary_results.csv`
- `evaluation_episodes.csv`
- `stability_summary.csv`
- `optuna_ppo_trials.csv`
- `optuna_dqn_trials.csv`
- `optuna_a2c_trials.csv`

E graficos como:

- curvas de aprendizado;
- recompensa media e minima durante treino;
- distribuicao das recompensas;
- comparacao final de reward, downlink e potencia;
- historico e importancia dos hiperparametros do Optuna.

## Observação

Este repositorio ja inclui o pacote `network/` necessario para rodar o ambiente. Nao e necessario clonar o repositorio antigo `rl-wireless` para executar este projeto.
