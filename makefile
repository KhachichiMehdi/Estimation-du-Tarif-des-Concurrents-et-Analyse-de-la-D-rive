# Makefile - Automatisation locale

# Variables
POETRY_RUN=poetry run

# 📦 Installation
install:
	poetry install

# 🧪 Tests unitaires
test:
	$(POETRY_RUN) pytest --disable-warnings --maxfail=1

# 📊 Couverture de test
coverage:
	$(POETRY_RUN) pytest --cov=src --cov-report=term --cov-report=html

# 🧼 Formatage
format:
	$(POETRY_RUN) black src
	$(POETRY_RUN) isort src

# 🔍 Lint
lint:
	$(POETRY_RUN) ruff check src
	$(POETRY_RUN) mypy src

# ✅ Pre-commit
precommit:
	$(POETRY_RUN) pre-commit run --all-files


# 💥 Tout lancer (sauf docs)
all: install lint test precommit
