.PHONY: install-dev clean format check install uninstall test diff-test


format:
	black algo
	blackdoc algo
	isort algo

check:
	black algo --check --diff
	blackdoc algo --check
	flake8 --config pyproject.toml --ignore E203,E501,W503,E741 algo
	mypy --config pyproject.toml algo
	isort algo --check --diff
	