#!/bin/bash

OS=$1

if [[ "$OS" == "ubuntu-latest" ]]
then
  LATEST_VERSION=$(ls -1 /opt/intel/oneapi/compiler/ | grep -v latest | sort | tail -1)
  source /opt/intel/oneapi/compiler/"$LATEST_VERSION"/env/vars.sh
else
  source /opt/intel/oneapi/setvars.sh
fi
