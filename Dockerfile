FROM python:3.12-slim

WORKDIR /app
 
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt
 
COPY . .
 
# Build FAISS index during image build

RUN python ingest.py
 
EXPOSE 8080
EXPOSE 8501

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
 
CMD ["streamlit", "run", "streamlit.py", "--server.port=8501", "--server.address=0.0.0.0"]
