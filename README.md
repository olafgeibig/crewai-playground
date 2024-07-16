poetry self add poetry-plugin-mono-repo-deps
poetry self add 'poethepoet[poetry_plugin]'
poetry config settings.virtualenvs.in-project true
poetry install