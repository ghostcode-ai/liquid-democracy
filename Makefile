.PHONY: setup test sim sweep dashboard clean help

PYTHON ?= python3

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

setup: ## Install project + dev dependencies
	$(PYTHON) -m pip install -e ".[dev]"
	@echo ""
	@echo "\033[32mSetup complete.\033[0m Run 'make test' to verify."

test: ## Run all tests
	$(PYTHON) -m pytest tests/ -v --tb=short

test-cov: ## Run tests with coverage report
	$(PYTHON) -m pytest tests/ -v --cov=agents --cov=engine --cov=tally --cov=scenarios --tb=short

sim: ## Run simulation (interactive mode — prompts for options)
	$(PYTHON) scripts/run_simulation.py --interactive

sim-quick: ## Run baseline with 500 agents (fast sanity check)
	$(PYTHON) scripts/run_simulation.py --agents 500 --scenario baseline

sim-full: ## Run baseline with 10,000 agents
	$(PYTHON) scripts/run_simulation.py --agents 10000 --scenario baseline

sweep: ## Run parameter sensitivity sweep (1000 agents)
	$(PYTHON) scripts/parameter_sweep.py --agents 1000 --output sweep_results.csv

sweep-fast: ## Run parameter sweep with 300 agents (faster)
	$(PYTHON) scripts/parameter_sweep.py --agents 300 --output sweep_results.csv

dashboard: ## Launch the Streamlit dashboard
	@echo "\033[36mStarting dashboard at http://localhost:8501\033[0m"
	$(PYTHON) -m streamlit run dashboard/app.py

dashboard-debug: ## Launch dashboard with LLM debug logging (prints req/resp to stderr)
	@echo "\033[33mDebug mode: Claude req/resp pairs will print to stderr\033[0m"
	LLM_DEBUG=1 $(PYTHON) -m streamlit run dashboard/app.py

clean: ## Remove build artifacts and caches
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache build dist *.egg-info
	rm -f sweep_results.csv sweep_results.json
