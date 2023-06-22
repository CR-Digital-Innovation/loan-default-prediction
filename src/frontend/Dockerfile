# -------------- BUILD STAGE ------------------
# Pull base image of python 3.10 slim version
FROM python:3.10-slim as base
# WorkDir 
WORKDIR /app
# Copy requirements.txt file
COPY frontend/requirements.txt .
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
# Copy UI Files to app directory
COPY frontend/*.py /app/
# Define the Environment Variable for Backend API Host URL
ENV BACKEND_HOST_URL="http://localhost:8000"
# Expose default Streamlit UI port
EXPOSE 8501
# Run streamlit
CMD ["streamlit", "run", "HomePage.py"]