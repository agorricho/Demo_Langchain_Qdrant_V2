ource .env
## update Dec 1, 18:35

function container_command() {
    if [ -n "`command -v docker`" ]; then
        echo "docker"
        return 0 
    elif [ -n "`command -v podman`" ]; then
        echo "podman"
        return 0 
    else
        echo "Neither 'docker' nor 'podman' found." >> /dev/stderr
        echo "echo"
        return -1
    fi
}

COMMAND=$(container_command)

export OLLAMA_CONTAINER_IMAGE="docker.io/ollama/ollama:latest"

function ollama_start() { # SERVER_NUM, GPUS, OLLAMA_STAGING_DIR, OLLAMA_HOST_NAME, OLLAMA_PORT, OLLAMA_CONTAINER_IMAGE
    SERVER_NUM=$1
    GPUS=$2
    OLLAMA_STAGING_DIR=$3
    OLLAMA_DIR="${OLLAMA_STAGING_DIR}/${SERVER_NUM}"
    OLLAMA_HOST_NAME=10.230.100.240
    OLLAMA_PORT=17434
    OLLAMA_CONTAINER_IMAGE=$6

    my_port=$(($OLLAMA_PORT + $SERVER_NUM - 1))
    my_name="${OLLAMA_HOST_NAME}_${SERVER_NUM}"


    if [ "${GPUS}" == "none" ]; then
        echo "Using CPU only"
        DEVICE=""
    else
        if [ "${GPUS}" == "all" ]; then
            GPUS=`nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd ","`
        fi
        echo "Using GPU(s): ${GPUS}"
        DEVICE=`echo "${GPUS}" | tr ',' '\n' | while read N; do echo -n " --device nvidia.com/gpu=$N "; done`
        DEVICE="${DEVICE} --security-opt=label=disable"
    fi
    mkdir -p $OLLAMA_DIR
    
    export OLLAMA_HOST=10.230.100.240
 
    ${COMMAND} run -d \
        ${DEVICE} \
        --name "${my_name}" \
        --hostname "${my_name}" \
        -p ${my_port}:11434 \
        -e OLLAMA_HOST \
        -v $PWD:/workspace/host \
        -v ${OLLAMA_DIR}:/root/.ollama \
        --restart always \
            "${OLLAMA_CONTAINER_IMAGE}"

}

ACTION=$1

echo "ACTION: $ACTION"


case $ACTION in
    pull)
        echo "Pulling containers"
        $COMMAND pull ${OLLAMA_CONTAINER_IMAGE}
        # $COMMAND pull ${NGINX_CONTAINER_IMAGE}
        # $COMMAND pull ${AlPINE_CONTAINER_IAMGE}
        ;;

    start)
        ## Create network
        # $COMMAND network create ${PRIVATE_NETWORK_NAME}

        ## Start Ollama servers
        seq 1 $NUMBER_OLLAMA_INSTANCES | while read N
        do
            echo "Starting Ollama server #${N}"
            # SERVER_NUM, GPUS, OLLAMA_STAGING_DIR, OLLAMA_HOST_NAME, PRIVATE_NETWORK_NAME, OLLAMA_CONTAINER_IMAGE
            # SERVER_NUM=$1
            ollama_start $N $GPUS \
                $OLLAMA_STAGING_DIR \
                $OLLAMA_HOST_NAME \
                $OLLAMA_PORT \
                $OLLAMA_CONTAINER_IMAGE
            echo "Ollama server #${N} started"
        done
        ;;

    stop)
        echo "Stopping containers"
        seq 1 $NUMBER_OLLAMA_INSTANCES | while read N
        do
            CONTAINER_NAME="${OLLAMA_HOST_NAME}_${N}"
            echo "Stopping $CONTAINER_NAME"
            $COMMAND stop -i $CONTAINER_NAME
            $COMMAND rm -i $CONTAINER_NAME
        done
        ;;

    models)
        echo "Pulling Ollama models"
        seq 1 $NUMBER_OLLAMA_INSTANCES | while read N
        do
            CONTAINER_NAME="${OLLAMA_HOST_NAME}_${N}"
            pscrpt=`cat ./ollama-models.txt | grep -v '^#' | while read M X; do if [ -n "$M" ]; then echo "ollama pull $M;"; fi done`
            $COMMAND exec -i $CONTAINER_NAME bash -c "$pscrpt"
        done
        ;;

    status)
        echo "Status of containers"
        $COMMAND ps
        ;;

    environment)
        echo "Verifying Environment"
        echo '==================================================================================='
        printenv | sort
        echo '==================================================================================='
        ;;

esac
