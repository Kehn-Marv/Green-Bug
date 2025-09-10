.PHONY: help redis-up install-backend test-redis test-all start-backend start-frontend

help:
	@echo "Available targets:"
	@echo "  redis-up            Start redis via docker-compose"
	@echo "  install-backend     pip install -e backend"
	@echo "  test-redis          Run Redis-backed tests"
	@echo "  test-all            Run full backend test suite"
	@echo "  start-backend       Start backend (uvicorn)"
	@echo "  start-frontend      Start frontend dev server (npm)"

redis-up:
	docker compose up -d

install-backend:
	python -m pip install -e backend

test-redis:
	# Ensure REDIS_URL points at local Redis
	set REDIS_URL=redis://127.0.0.1:6379/0 && python -m pytest backend/tests/test_redis_job_store.py -q

test-all:
	python -m pytest backend -q

start-backend:
	python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8080 --reload

start-frontend:
	npm run --prefix frontend dev
