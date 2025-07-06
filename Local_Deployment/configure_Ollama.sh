#!/bin/bash

# Configuration Parameters
export OLLAMA_CONTAINER_IMAGE="docker.io/ollama/ollama:latest"
export NGINX_CONTAINER_IMAGE="docker.io/library/nginx:latest"
export ALPINE_CONTAINER_IAMGE="docker.io/library/alpine:latest"

export OLLAMA_HOST_NAME="${OLLAMA_HOST_NAME:-ollama}"
export NGINX_HOST_NAME="${NGINX_HOST_NAME:-nginx}"
export ALPINE_HOST_NAME="${ALPINE_HOST_NAME:-alpine}"

export PRIVATE_NETWORK_NAME="${PRIVATE_NETWORK_NAME:-privatenet}"

# Check if required environment variables are set
if [ -z "PROXY_PORT" ]; then
    echo "PROXY_PORT is not set"
    exit 1
fi
if [ -z "BASTION_PORT" ]; then
    echo "BASTION_PORT is not set"
    exit 1
fi
if [ -z "OLLAMA_PORT" ]; then
    echo "OLLAMA_PORT is not set"
    exit 1
fi
if [ -z "GPUS" ]; then
    echo "GPUS is not set. Example: GPUS=\"0,1,2,3\""
    exit 1
fi
if [ -z "NUMBER_OLLAMA_INSTANCES" ]; then
    echo "NUMBER_OLLAMA_INSTANCES is not set"
    exit 1
fi
if [ -z "GENAI_TUTOR_SECRET" ]; then
    echo "GENAI_TUTOR_SECRET is not set"
    exit 1
    # Example: GENAI_TUTOR_SECRET="genai-tutor-secret"
fi
if [ -z "OLLAMA_STAGING_DIR" ]; then
    echo "OLLAMA_STAGING_DIR is not set"
    exit 1
fi
if [ -z "WORK_DIR" ]; then
    echo "WORK_DIR is not set"
    exit 1
fi

cat /dev/null > .env

# Append additional environment variables
echo "export PROXY_PORT=${PROXY_PORT}" >> .env
echo "export BASTION_PORT=${BASTION_PORT}" >> .env
echo "export OLLAMA_PORT=${OLLAMA_PORT}" >> .env
echo "export GPUS=${GPUS}" >> .env
echo "export NUMBER_OLLAMA_INSTANCES=${NUMBER_OLLAMA_INSTANCES}" >> .env
echo "export GENAI_TUTOR_SECRET=${GENAI_TUTOR_SECRET}" >> .env
echo "export OLLAMA_STAGING_DIR=${OLLAMA_STAGING_DIR}" >> .env
echo "export WORK_DIR=${WORK_DIR}" >> .env

echo "export OLLAMA_CONTAINER_IMAGE=${OLLAMA_CONTAINER_IMAGE}" >> .env
echo "export NGINX_CONTAINER_IMAGE=${NGINX_CONTAINER_IMAGE}" >> .env
echo "export ALPINE_CONTAINER_IAMGE=${ALPINE_CONTAINER_IAMGE}" >> .env
echo "export OLLAMA_HOST_NAME=${OLLAMA_HOST_NAME}" >> .env
echo "export NGINX_HOST_NAME=${NGINX_HOST_NAME}" >> .env
echo "export ALPINE_HOST_NAME=${ALPINE_HOST_NAME}" >> .env
echo "export PRIVATE_NETWORK_NAME=${PRIVATE_NETWORK_NAME}" >> .env

# Create OLLAMA_PORT_#
seq 1 $NUMBER_OLLAMA_INSTANCES | while read i
do
    p=$(($OLLAMA_PORT + $i - 1))
    echo "export OLLAMA_PORT_${i}=$p" >> .env
    echo "export OLLAMA_HOST_NAME_${i}=${OLLAMA_HOST_NAME}${i}" >> .env
    echo "server ${OLLAMA_HOST_NAME}${i};" >> .nginx_backend_list
done

# Get secrets from AWS Secrets Manager
aws secretsmanager get-secret-value --secret-id ${GENAI_TUTOR_SECRET} \
| jq -r ".SecretString" | jq . \
| jq -r 'to_entries | .[] | "export \(.key | ascii_upcase)=\"\(.value)\""' \
>> .env
# source .secrets.env
# rm .secrets.env
echo "Secrets retrieved from AWS Secrets Manager"



# Update Nginx Configuration
cat /dev/null > .nginx_backend_list
seq 1 $NUMBER_OLLAMA_INSTANCES | while read i
do
    echo "server ${OLLAMA_HOST_NAME}${i};" >> .nginx_backend_list
done
export NGINX_OLLAMA_BACKEND_LIST="$(cat .nginx_backend_list | tr '\n' ' ')"
echo "export NGINX_OLLAMA_BACKEND_LIST=\"${NGINX_OLLAMA_BACKEND_LIST}\"" >> .env


