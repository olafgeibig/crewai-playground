# CrewAI-Playground
1. If activated env, then deactivate first
2. Run poetry install in the root and in all projects    

# Notes
Poetry setup
```
poetry self add poetry-plugin-mono-repo-deps
poetry self add 'poethepoet[poetry_plugin]'
poetry config virtualenvs.in-project true
```
# Use Cases
## After a lock file update
```
poetry lock --no-update
poetry install --no-root
```