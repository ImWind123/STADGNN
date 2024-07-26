software environment
python>=3.8
pip install torch==2.1.0
pip install numpy==1.26.2
pip install pandas==1.2.4

We use Python 3.9 to implement the proposed method, and all experiments are run on the machine of Windows system. To run the model, you need to put the dataset file in the STADGNN/data/ path, set the necessary hyperparameters, and run the python file train.py. We provide an interface for using GPU in train.py. If you have a GPU, it is recommended to use GPU for training.
