# Stage 1: Use an official Python runtime as a parent image
# Using a specific version ensures reproducibility
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY ./requirements.txt /app/requirements.txt

# Install any needed packages specified in requirements.txt
# --no-cache-dir makes the image smaller
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application's code into the container at /app
# This includes the /src folder and the /artifacts folder
COPY . /app

# Expose port 8000 to the outside world.
# This is the port uvicorn will run on.
EXPOSE 8000

# Command to run the application when the container launches.
# This is the same command you use to run the API locally.
CMD ["uvicorn", "src.predict:app", "--host", "0.0.0.0", "--port", "8000"]