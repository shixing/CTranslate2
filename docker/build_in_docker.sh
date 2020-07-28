CTRANSLATE2_ROOT=/opt/ctranslate2_prefix
cd /ctranslate2
rm -rf build
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX=${CTRANSLATE2_ROOT} \
      -DCMAKE_PREFIX_PATH="/root/thrust;/root/cub;/root/mkl-dnn" \
      -DWITH_CUDA=ON -DWITH_MKLDNN=ON -DCUDA_NVCC_FLAGS="-Xfatbin -compress-all" \
      -DCUDA_ARCH_LIST=Common -DCMAKE_BUILD_TYPE=Release .. && \
      VERBOSE=1 make -j4 && make install

cp /opt/intel/lib/intel64/libiomp5.so ${CTRANSLATE2_ROOT}/lib && \
    cp -P /root/mkl-dnn/lib/libmkldnn.so* ${CTRANSLATE2_ROOT}/lib && \
    cp -P /usr/lib/x86_64-linux-gnu/libcudnn.so* ${CTRANSLATE2_ROOT}/lib && \
    cp -P /usr/lib/x86_64-linux-gnu/libnvinfer.so* ${CTRANSLATE2_ROOT}/lib

