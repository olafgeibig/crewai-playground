poetry self add poetry-plugin-mono-repo-deps
poetry self add 'poethepoet[poetry_plugin]'
poetry config virtualenvs.in-project true
poetry install