.PHONY: install-dev clean format check install uninstall test diff-test


format:
	black algos
	blackdoc algos
	isort algos

check:
	black algos --check --diff
	blackdoc algos --check
	flake8 --config pyproject.toml --ignore E203,E501,W503,E741 algos
	mypy --config pyproject.toml algos
	isort algos --check --diff

push:
	git add .
	git commit -m "."
	git push -u origin HEAD
	