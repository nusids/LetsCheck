FROM python:3.7.1-alpine
WORKDIR /project
ADD . /project
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
CMD ["python3", "frontend.py"]
