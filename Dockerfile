FROM pypy:3


COPY ./src /app/src/
COPY ./data /app/data/
COPY ./requirements-pypy.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt
WORKDIR /app/src


EXPOSE 5000
CMD [ "pypy3", "./endpoint.py" ]
