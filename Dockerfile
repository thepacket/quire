# Quire on a small Debian image with the Maxima and FriCAS backends available.
FROM python:3.13-slim

RUN apt-get update \
 && apt-get install -y --no-install-recommends maxima fricas \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY quire ./quire
COPY modules ./modules
COPY examples ./examples
COPY bench ./bench
RUN pip install --no-cache-dir $(python -c "import tomllib; print(' '.join(tomllib.load(open('pyproject.toml','rb'))['project']['dependencies']))") \
 && python -c "import quire, sympy, numpy, scipy" \
 && python -c "from pathlib import Path; from quire.modules.registry import load_registry; r = load_registry([Path('modules')]); print([(m.name, m.error) for m in r.modules])"

ENV PORT=8080 \
    QUIRE_HOST=0.0.0.0 \
    QUIRE_WORKSHEETS=/data/worksheets \
    PYTHONUNBUFFERED=1
RUN mkdir -p /data/worksheets
VOLUME ["/data"]
EXPOSE 8080

CMD ["python", "-m", "quire", "--no-browser"]
