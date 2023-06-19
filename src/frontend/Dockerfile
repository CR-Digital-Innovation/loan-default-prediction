# Pull base image of python 3.10 slim version
FROM python:3.10-slim
# WorkDir 
WORKDIR /app
# Copy requirements.txt file
COPY requirements.txt HomePage.py /app/
# Install dependencies from requirements
RUN python -m pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt
# Define the Environment Variable for Backend API Host URL
ENV BACKEND_HOST_URL="http://localhost:8000"
# Expose default Streamlit UI port
EXPOSE 8501
# Run streamlit
CMD ["streamlit", "run", "HomePage.py"]