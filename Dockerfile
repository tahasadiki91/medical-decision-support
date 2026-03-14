# Use an official lightweight Python image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container
COPY . /app

# Install the required packages
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port Streamlit uses
EXPOSE 8501

# Command to run the Streamlit application
CMD ["streamlit", "run", "app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
