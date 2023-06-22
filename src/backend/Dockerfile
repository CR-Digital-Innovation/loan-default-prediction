# -------------- BUILD STAGE ------------------
# Pull base image of python 3.10 slim version
FROM python:3.10-slim as base
# WorkDir 
WORKDIR /app
# Copy requirements.txt file
COPY backend/requirements.txt .
# Install dependencies from requirements
RUN python -m venv .venv && .venv/bin/pip install --no-cache-dir -U pip setuptools
RUN .venv/bin/pip install --no-cache-dir -r requirements.txt && find /app/.venv \( -type d -a -name test -o -name tests \) -o \( -type f -a -name '*.pyc' -o -name '*.pyo' \) -exec rm -rf '{}' \+

# -------------- RUNNER STAGE -------------------
# Pull base image of python 3.10 slim version
FROM python:3.10-slim
# WorkDir 
WORKDIR /app
# Copy dependcies from base image
COPY --from=base /app /app
ENV PATH="/app/.venv/bin:$PATH"
# Copy backend files to the working directory
RUN mkdir backend utils
COPY backend/*.py /app/backend/
COPY backend/.env /app/
# Copy Utils folder to working directory
# RUN cd ../ && cp -r utils /app/
COPY utils/*.py /app/utils/
# Expose default port of FASTAPI
EXPOSE 8000
# Run FASTAPI Backend
CMD ["python", "backend/server.py"]