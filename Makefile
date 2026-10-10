.PHONY: install test eval eval-semantic serve ui stack prompt anonymise sec redteam lint metrics

install:
	pip install -e .[dev,ui,otel]
	npm ci

lint:
	ruff check src eval tests scripts

test:
	pytest -q

eval:            ## offline profile, gated
	python eval/run_eval.py

eval-semantic:   ## real embeddings + cross-encoder (needs pip install -e .[dense])
	RAGKIT_CONFIG=configs/semantic.yaml python eval/run_eval.py

serve:           ## API on :8000
	uvicorn --factory ragkit.api.app:get_app --port 8000 --reload

ui:              ## Streamlit app, in-process; add RAGKIT_API_URL=http://localhost:8000 for HTTP mode
	streamlit run src/ragkit/ui/streamlit_app.py

stack:           ## API + UI + Prometheus + Grafana (needs .env with PSEUDO_KEY, RAGKIT_ADMIN_TOKEN, GRAFANA_PASSWORD)
	docker compose up --build

metrics:         ## peek at the Prometheus exposition of a running API
	curl -s localhost:8000/metrics | grep '^ragkit_'

prompt:          ## show the exact prompt for a question: make prompt Q="..."
	python scripts/show_prompt.py "$(Q)"

anonymise:       ## before/after of a sample report: make anonymise R=lab_report_hematology
	python scripts/anonymise_report.py $(R)

sec:             ## deterministic promptfoo suite (service must be running)
	npx promptfoo eval -c promptfoo/promptfooconfig.yaml --no-cache

redteam:         ## generative red-team (needs ANTHROPIC_API_KEY)
	npx promptfoo redteam run -c promptfoo/redteam.yaml
