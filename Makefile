# Hydra v5 — paper trader + dashboard
# Requires: Python 3 with deps from requirements.txt (see `make deps`)
#
# Quick start:
#   make live              # dashboard + Kraken websocket (loads params per config logic)
#   make live-relaxed      # force hydra_relaxed.json
#   make replay REPLAY=runs/your.jsonl

PYTHON       ?= python3
RECORD       ?= runs/live_$(shell date +%Y-%m-%d_%H%M).jsonl
REPLAY       ?=
SPEED        ?= 10
PARAMS       ?= hydra_relaxed.json
HIST_SYMBOLS ?= BTC/USD,ETH/USD,SOL/USD
HIST_HOURS   ?= 12
OPTUNA_RECORDINGS ?=
OPTUNA_TRIALS     ?= 100

.PHONY: help deps live live-relaxed live-defaults live-record \
	live-headless live-headless-record replay replay-fast backtest historical optuna

help: ## Show this help
	@echo "Hydra v5 — targets:"
	@grep -hE '^[a-zA-Z0-9_.-]+:.*##' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*##"}; {printf "  %-18s %s\n", $$1, $$2}'
	@echo ""
	@echo "Examples:"
	@echo "  make live"
	@echo "  make live-headless              # unattended live (no GUI)"
	@echo "  make live-headless-record       # same + default RECORD path"
	@echo "  make replay REPLAY=runs/session.jsonl SPEED=5"
	@echo "  make backtest REPLAY=runs/session.jsonl"
	@echo "  make optuna OPTUNA_RECORDINGS=runs/a.jsonl,runs/b.jsonl"

deps: ## Install Python dependencies (pip install -r requirements.txt)
	$(PYTHON) -m pip install -r requirements.txt

live: ## Live dashboard + Kraken WS (default param loading: hydra_best / relaxed fallback)
	$(PYTHON) master.py

live-relaxed: ## Live with explicit hydra_relaxed.json
	$(PYTHON) master.py --params $(PARAMS)

live-defaults: ## Live with built-in Config defaults only
	$(PYTHON) master.py --no-params

live-record: ## Live + append all WS messages to RECORD (default: runs/live_YYYY-mm-dd_HHMM.jsonl)
	$(PYTHON) master.py --record $(RECORD)

live-headless: ## Live without dashboard (daemon-style; Ctrl+C prints summary)
	$(PYTHON) master.py --no-dashboard

live-headless-record: ## Live headless + RECORD
	$(PYTHON) master.py --no-dashboard --record $(RECORD)

replay: ## Replay a JSONL recording in the dashboard (requires REPLAY=path)
	@test -n "$(REPLAY)" || (echo "Set REPLAY, e.g. make replay REPLAY=runs/day.jsonl"; exit 1)
	$(PYTHON) master.py --replay $(REPLAY) --speed $(SPEED)

replay-fast: ## Replay at max CPU speed (requires REPLAY=)
	@test -n "$(REPLAY)" || (echo "Set REPLAY=..."; exit 1)
	$(PYTHON) master.py --replay $(REPLAY) --speed inf

backtest: ## Headless full-signal replay backtest (requires REPLAY=)
	@test -n "$(REPLAY)" || (echo "Set REPLAY=..."; exit 1)
	$(PYTHON) master.py --replay $(REPLAY) --backtest

historical: ## Kraken REST OHLC-only historical run (no dashboard)
	$(PYTHON) master.py --historical-backtest --hist-symbols $(HIST_SYMBOLS) --hist-hours $(HIST_HOURS)

optuna: ## Headless Optuna on recordings (set OPTUNA_RECORDINGS=comma-separated paths)
	@test -n "$(OPTUNA_RECORDINGS)" || (echo "Set OPTUNA_RECORDINGS=runs/a.jsonl or list"; exit 1)
	$(PYTHON) master.py --optuna --optuna-recordings $(OPTUNA_RECORDINGS) --optuna-trials $(OPTUNA_TRIALS) --no-params
