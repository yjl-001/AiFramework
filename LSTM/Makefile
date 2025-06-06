# Makefile for the LSTM from Scratch project

# Define the Python interpreter
PYTHON = python3

# Define the main script
MAIN_SCRIPT = main.py

# Default target: run the main script
run:
	@echo "Running the main application..."
	$(PYTHON) $(MAIN_SCRIPT)

# Target to run sequence classification example (if main.py supports arguments or separate scripts)
run-classification:
	@echo "Running Sequence Classification Example..."
	# Modify main.py to accept an argument or have a separate script for this
	$(PYTHON) $(MAIN_SCRIPT) # Assuming main.py runs classification by default or can be configured

# Target to run NMT example
run-nmt:
	@echo "Running NMT Seq2Seq Example..."
	# Modify main.py to accept an argument to run NMT, e.g., $(PYTHON) $(MAIN_SCRIPT) nmt
	# For now, you might need to uncomment the NMT part in main.py manually
	$(PYTHON) $(MAIN_SCRIPT) # Placeholder: Adjust if main.py needs specific args for NMT

# Target to install dependencies
install:
	@echo "Installing dependencies from requirements.txt..."
	$(PYTHON) -m pip install -r requirements.txt

# Target to run tests (if tests are added)
# test:
# 	@echo "Running tests..."
# 	$(PYTHON) -m unittest discover -s tests -p '*_test.py'

# Target to clean up bytecode files and other generated files
clean:
	@echo "Cleaning up..."
	find . -type f -name '*.pyc' -delete
	find . -type d -name '__pycache__' -exec rm -rf {} +
	@echo "Cleanup complete."

# Phony targets (targets that don't represent actual files)
.PHONY: run run-classification run-nmt install test clean