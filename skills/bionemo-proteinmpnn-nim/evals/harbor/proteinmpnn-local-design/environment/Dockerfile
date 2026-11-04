FROM nvcr.io/nim/ipd/proteinmpnn:latest

USER root

RUN mkdir -p \
      /workspace/input \
      /workspace/output \
      /logs/agent \
      /logs/verifier \
    && chown -R 1000:1000 /workspace /logs

COPY --chown=1000:1000 input/ /workspace/input/

USER nvs:1000
WORKDIR /workspace
