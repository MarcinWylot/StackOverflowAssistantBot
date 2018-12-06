FROM python:3

RUN pip install --no-cache-dir requests nltk chatterbot sklearn gensim
RUN python -m nltk.downloader -d /nltk_data  punkt stopwords

WORKDIR /app
COPY app ./
RUN chmod o+r *

USER nobody

CMD [ "python", "./main.py" ]
