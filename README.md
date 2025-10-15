# CS 272 Public Repo

## Environment setup for Gymnasium and stable-baselines3
On your host machine (your laptop, not a container), you need to have ```tensorboard```.
```bash
pip install tensorboard
```

You are going to run the program on a Docker container. [Install Docker](https://www.docker.com/) to your laptop.
Then, create an image named cs272demo.
```bash
docker build -t cs272demo .
```

```bash
# mkdir logs
# mkdir models
docker run --rm -v ./logs:/workspace/logs -v ./models:/workspace/models cs272demo intro-to-sb3/sample-sb.py
```

You can visualize the learning results on tensorboard by specifying the log directory mounted. You can run the following command on your host machine.
```bash
tensorboard --logdir ./logs/
```

<!-- 
When you want to run ```intro-to-gym/sample-cp.py```, you can use the following command.
```bash
docker run --rm --shm-size=1.65gb cs272demo intro-to-gym/sample-cp.py
```
Note: ```--shm-size=1.65gb``` specified the amount of the shared memory. 1.65GB is recommended by rllib.

When you want to run a program with ```rllib```, you need to mount the ray result directory on your host machine with the ```-v``` option.
```bash
docker run --rm --shm-size=1.65gb -v .:/root/ray_results cs272demo lec-rllib/first_rllib.py
```
-->