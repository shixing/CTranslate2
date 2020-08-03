export CTRANSLATE2_ROOT=/opt/ctranslate2_prefix
export LD_LIBRARY_PATH=${CTRANSLATE2_ROOT}/lib

cd /ctranslate2/python
python3 setup.py build
PYTHONIOENCODING=utf-8 PYTHONPATH=/ctranslate2/python/build/lib.linux-x86_64-3.5 python3 /ctranslate2/docker/test_decode_with_fsa_prefix.py