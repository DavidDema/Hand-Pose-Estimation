#sudo docker container stop hpe_ros
#sudo docker container rm hpe_ros
sudo docker build -t hpe_ros .

sudo docker run -it --network=host hpe_ros
#thispid=$(sudo docker run --network=host --name=hpe_ros -t -d -v)
#sudo docker exec -it hpe_ros /bin/bash
