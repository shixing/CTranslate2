#!/bin/bash
#
# Build latest:
# ./build_all.sh
#
# Build and push version X.Y.Z:
# ./build_all.sh X.Y.Z 1

set -e

# allow user to run this script from anywhere
# from https://stackoverflow.com/a/246128
# one-liner which will give you the full directory name
# of the script no matter where it is being called from
unset CDPATH
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

ROOT_DIR=$DIR/..
cd $ROOT_DIR

VERSION=${1:-latest}
PUSH=${2:-0}
IMAGE=opennmt/ctranslate2

build()
{
    PLAT=$1
    VERSION=$3

    if [ "$#" -eq 3 ]; then
        UBUNTU_VERSION_ARG="--build-arg UBUNTU_VERSION=$2"
        UBUNTU_MAJOR_VERSION="${2%.*}"
        TAG_SUFFIX="${PLAT}-${UBUNTU_MAJOR_VERSION}"
    else
        UBUNTU_VERSION_ARG=""
        TAG_SUFFIX=$PLAT
    fi

    LATEST=$IMAGE:latest-$TAG_SUFFIX
    TAGGED=$IMAGE:$VERSION-$TAG_SUFFIX
    docker build --pull $UBUNTU_VERSION_ARG -t $TAGGED -f docker/Dockerfile.$PLAT .
    if [ $PUSH -eq 1 ]; then
        docker push $LATEST
    fi
    if [ "$TAGGED" != "$LATEST" ]; then
        docker tag $LATEST $TAGGED
        if [ $PUSH -eq 1 ]; then
            docker push $TAGGED
        fi
    fi
}

pwd
build ubuntu-dev-gpu 16.04 1.4.0
