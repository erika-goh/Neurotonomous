VENV = venv
PYTHON = $(VENV)/bin/python
PIP = $(VENV)/bin/pip

.PHONY: run
run: $(VENV)/bin/activate
	$(PYTHON) Neurotonomous.py

$(VENV)/bin/activate: requirements.txt
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@touch $(VENV)/bin/activate  # Update timestamp so make knows it's done

.PHONY: install
install: $(VENV)/bin/activate

.PHONY: clean
clean:
	rm -rf $(VENV)
	find . -type f -name '*.pyc' -delete
	find . -type d -name '__pycache__' -delete

.PHONY: quick
quick: $(VENV)/bin/activate
	sed -i '' 's/max_episodes = [0-9]*/max_episodes = 50/' Neurotonomous.py || sed -i 's/max_episodes = [0-9]*/max_episodes = 50/' Neurotonomous.py
	$(PYTHON) Neurotonomous.py

.PHONY: help
help:
	@echo "Available commands:"
	@echo "  make          - Run Neurotonomous (default)"
	@echo "  make install  - Create virtual environment and install dependencies"
	@echo "  make run      - Run the simulator"
	@echo "  make quick    - Temporarily set to 50 episodes and run (for fast testing)"
	@echo "  make clean    - Remove virtual environment and cache files"
	@echo "  make help     - Show this help"
