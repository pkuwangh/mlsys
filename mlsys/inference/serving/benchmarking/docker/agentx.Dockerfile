ARG BASE_IMAGE=python:3.12-slim
FROM ${BASE_IMAGE}
ARG AGENTX_VERSION
COPY docker/install-agentx.sh /opt/install-agentx.sh
RUN bash /opt/install-agentx.sh "${AGENTX_VERSION}"
ENTRYPOINT []
CMD ["agentx", "--help"]
