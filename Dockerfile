FROM pypy:3


COPY ./src /app/src/
COPY ./data /app/data/
COPY ./requirements.txt /app
RUN pip install -r /app/requirements.txt
WORKDIR /app/src


EXPOSE 5000
CMD [ "pypy3", "./endpoint.py" ]
