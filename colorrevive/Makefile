# ColorRevive — developer workflow shortcuts
# Run from the colorrevive/ directory.

.PHONY: help install install-backend install-frontend dev backend frontend \
        test test-backend test-frontend lint format docker-up docker-down docker-build clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

install: install-backend install-frontend ## Install everything

install-backend: ## Install Python deps (activate a venv first)
	cd backend && pip install -r requirements.txt && pip install pytest httpx

install-frontend: ## Install Node deps
	cd frontend && npm ci

dev: ## Run backend + frontend concurrently (needs two terminals or tmux)
	@echo "Run 'make backend' and 'make frontend' in separate terminals."

backend: ## Start FastAPI on :8000 with reload
	cd backend && uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

frontend: ## Start Next.js dev server on :3000
	cd frontend && npm run dev

test: test-backend test-frontend ## Run all tests

test-backend: ## Pytest suite (API, model, preprocessing)
	cd backend && python -m pytest -q

test-frontend: ## Vitest + React Testing Library suite
	cd frontend && npm test -- --run

lint: ## ESLint (frontend) + optional ruff/pyflakes (backend)
	cd frontend && npx next lint
	cd backend && python -m pyflakes app tests || true

format: ## Prettier for frontend sources
	cd frontend && npx prettier --write "{app,components,lib,tests}/**/*.{ts,tsx,css}"

docker-build: ## Build both images
	docker compose build

docker-up: ## Build & start API (:8000) + web (:3000)
	docker compose up --build -d

docker-down: ## Stop and remove containers
	docker compose down

clean: ## Remove build artifacts, caches, temp files
	rm -rf frontend/.next frontend/node_modules/.cache backend/.pytest_cache
	find backend -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
	rm -rf backend/tmp_uploads backend/storage
