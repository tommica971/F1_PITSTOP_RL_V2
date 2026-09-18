# =============================================================================
# F1_PITSTOP_RL — image CPU, Python 3.12
# =============================================================================
# Deux cibles :
#   service  (defaut)  : inference. Lance check_models.py : preuve que le
#                        modele livre se recharge dans l'image.
#   notebook           : service + Jupyter, pour la demonstration.
#
#   docker build -t f1-pitstop-rl .                       -> service
#   docker build --target notebook -t f1-pitstop-rl:nb .  -> notebook
#
# Versions : requirements.txt, alignees sur system_info.txt des modeles.
# CPU uniquement : reproductibilite (device='cpu') et image legere.
# =============================================================================

FROM python:3.12.10-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# libgomp : runtime OpenMP de torch CPU. Pas de build-essential : toutes les
# dependances ont des wheels precompiles pour cp312 / linux x86_64.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependances d'abord : couche mise en cache tant que requirements.txt ne
# change pas. L'index CPU de PyTorch est declare dans le fichier lui-meme.
COPY requirements.txt .
RUN pip install -r requirements.txt

# Code et modele LIVRE uniquement. Les 19 autres modeles sont des artefacts
# d'experimentation ; pour les evaluer, monter ./models en volume
# (cf. docker-compose.yml, service train).
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY check_models.py .
COPY models/a2c/a2c_v2env_s2.zip ./models/a2c/a2c_v2env_s2.zip

# Le projet importe ses modules par leur nom court (f1_pitstop_env...).
ENV PYTHONPATH=/app/src/f1_pitstop_rl/env:/app/src/f1_pitstop_rl/config:/app/src/f1_pitstop_rl/data:/app/src/f1_pitstop_rl/training

# Utilisateur non root
RUN useradd --create-home --uid 1000 app && chown -R app:app /app
USER app


# --- Cible notebook -----------------------------------------------------------
FROM base AS notebook

USER root
RUN pip install notebook
COPY notebooks/ ./notebooks/
RUN chown -R app:app /app/notebooks
USER app

EXPOSE 8888
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", \
     "--ServerApp.token=", "--ServerApp.root_dir=/app"]


# --- Cible service (derniere = cible par defaut) -----------------------------
FROM base AS service

CMD ["python", "check_models.py"]
