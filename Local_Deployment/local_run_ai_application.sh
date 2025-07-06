# Define variables
OLLAMA_BASE_URL="http://10.230.100.240:17030"
ENDPOINT_PORT_NUMBER=8000

# check if podman or docker command is installed, define variable CONTAINERCMD
if [ -x "$(command -v podman)" ]; then
    CONTAINER_CMD=podman
    echo "podman found"
elif [ -x "$(command -v docker)" ]; then
    CONTAINER_CMD=docker
    echo "docker found"
else
    echo "Neither podman nor docker command found. Please install one of them."
    exit 1
fi

# Build container 
echo "Building container"
if [ -z "$CI_ENVIRONMENT_SLUG" ]; then
    CONTAINER_NAME="local/${USER}/ai-application"
else
    CONTAINER_NAME="local/${USER}/ai-application-${CI_ENVIRONMENT_SLUG}"
fi

$CONTAINER_CMD build -t $CONTAINER_NAME .

# Run container
echo "Running container"
$CONTAINER_CMD run -it --rm \
    -p ${ENDPOINT_PORT_NUMBER}:8000 \
    -e OLLAMA_BASE_URL=${OLLAMA_BASE_URL} \
    $CONTAINER_NAME



