## Setup building environment
```bash
sudo docker run -d --name ctranslate2 --gpus 1 --entrypoint sleep -p 8022:22 -v $PWD:/data opennmt/ctranslate2:1.4.0-ubuntu16-gpu 1d
```
