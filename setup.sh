# create venv
sudo apt install python3-venv
python3 -m venv .env
source .env/bin/activate
pip install -r requirements.txt

# install os packages 
sudo apt install gcc make

# install nvidia driver
wget https://us.download.nvidia.com/tesla/460.106.00/NVIDIA-Linux-x86_64-460.106.00.run
sudo bash NVIDIA-Linux-x86_64-460.106.00.run
