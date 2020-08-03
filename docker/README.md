# Setup a debug environment
## Build the docker 
```bash
cd docker
bash build_dev.sh
```
After building the docker, the docker will have all needed packages under `/root/`

## Start the docker 
```bash
cd docker
bash run_dev.sh
```
We will have two folder mapped into docker: 
- `/data`, 
    - where you should prepare the exported ctranslate2 model `export_ct2`
    - `source.txt`, the input source file.
    - `prefix_info.txt`, the prefix_info file. 
- `/ctranslate2`, the source code of CTranslate2. 

## Change code and debug
Config in CLion

NOTE: CLion will upload and build the code in "/tmp/XX/" folder


## Change the python code
First, compile the code in docker 
```
bash build_in_docker.sh
```
NOTE: we will compile the code in `/ctranslate2/build` and install the code into `/opt/ctranslate2_prefix`

Second, compile the python wrapper and test the python code
```
bash build_python.sh
```
NOTE: we will build the python code in `/ctranslate2/python/build`

# Install the compiled ctranslate2 to your local machine

## Install the lib

1. Copy `/opt/ctranslate2_prefix` from docker to your local machine, for example at '/home/ctranslate2'

2. Set the following environment variable. 
```bash
export PATH=$PATH:/home/ctranslate2
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/home/ctranslate2/lib
```

## Install the python wrapper

On your local machine
```bash
pip uninstall ctranslate2
cd python;
python setup.py install
```

## Test python code
```bash
cd docker
# You need to change the corresping path in test_decode_with_fsa_prefix_luban.py.
python test_decode_with_fsa_prefix_luban.py
```
