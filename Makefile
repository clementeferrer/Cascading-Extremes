API_PORT ?= 8000
WEB_PORT ?= 5173
IMMERSIVE_DIR := web/immersive
API_DIR := web/api

.PHONY: dev build serve

dev:
	@echo "Starting API on :$(API_PORT) and frontend on :$(WEB_PORT)"
	@(
		cd $(API_DIR) && uvicorn main:app --reload --reload-dir $(API_DIR) --port $(API_PORT)
	) & \
	(
		cd $(IMMERSIVE_DIR) && npm run dev -- --port $(WEB_PORT)
	) & \
	wait

build:
	@(cd $(IMMERSIVE_DIR) && npm run build)

serve:
	@IMMERSIVE_STATIC=$(IMMERSIVE_DIR)/dist uvicorn web.api.main:app --host 0.0.0.0 --port $(API_PORT)
