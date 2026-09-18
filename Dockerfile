# F1_PITSTOP_RL -- image CPU-only, reproductibilite prioritaire sur la vitesse
# (coherent avec la contrainte device='cpu' obligatoire pour l'entrainement/eval,
#  cf. memoire projet : le GPU introduit du non-determinisme malgre seed fixe)

FROM python:3.12-slim AS base

# Dependances systeme minimales : build-essential pour les wheels qui n'ont pas
# de binaire precompile pour cette combinaison python/arch, libgomp pour torch CPU
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copier uniquement requirements.txt d'abord pour profiter du cache Docker
# (le rebuild de l'image ne re-telecharge pas les dependances si le code change
#  sans que les dependances changent)
COPY requirements.txt .

# --index-url CPU pour torch : evite de telecharger les wheels CUDA (plusieurs Go
# inutiles ici puisque l'entrainement/eval tourne exclusivement en CPU dans ce projet)
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements.txt

# Code source, config, modeles entraines, notebooks
COPY src/ ./src/
COPY models/ ./models/
COPY notebooks/ ./notebooks/

# Donnees : volontairement PAS copiees dans l'image (montees en volume au lancement,
# cf. docker-compose.yml) -- eviter de figer des donnees dans l'image alourdit le
# build et complique la mise a jour du pool de GP sans rebuild
# COPY data/ ./data/

ENV PYTHONPATH=/app/src/f1_pitstop_rl/env:/app/src/f1_pitstop_rl/config:$PYTHONPATH
ENV PYTHONUNBUFFERED=1

# Port par defaut si Jupyter est lance pour ouvrir/executer les notebooks dans le
# conteneur (cf. docker-compose.yml, service "notebook")
EXPOSE 8888

CMD ["python", "-m", "f1_pitstop_rl.training.train_a2c", "--help"]
