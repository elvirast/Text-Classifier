Install `poetry`:

```shell script
curl -sSL https://raw.githubusercontent.com/sdispater/poetry/master/get-poetry.py | python - --version 1.0.0
. ~/.poetry/env
```

Install python library dependencies:

```shell script
poetry install
```


Run the service:

```shell script
poetry run python job_simulation/service.py
```

Open:

http://localhost:8010
