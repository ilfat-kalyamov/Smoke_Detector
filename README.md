
venv/Scripts/activate



sudo apt update
sudo apt upgrade
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt update
sudo apt install python3.11
sudo apt install python3.11-venv
python3.11 --version

sudo apt install python3-pip
python3 -m pip install --upgrade pip
pip --version

cd Smoke_Detector
python3.11 -m venv venv
source venv/bin/activate


pip install -r requirements.txt

pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

python3.11 main.py

deactivate