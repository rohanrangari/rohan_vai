DATA_DIR = r"C:\Rohan\Vector AI\repo\rohan_vai\src\data"
TRAIN_DATA_IMGS = rf"{DATA_DIR}\train-images-idx3-ubyte.gz"
TRAIN_DATA_LABELS = rf"{DATA_DIR}\train-labels-idx1-ubyte.gz"
TEST_DATA_IMGS = rf"{DATA_DIR}\test-images-idx3-ubyte.gz"
TEST_DATA_LABELS = rf"{DATA_DIR}\test-labels-idx1-ubyte.gz"
MODEL_SAVE_PATH = rf"{DATA_DIR}\models\trained_models"
REPORT_PATH = rf"{DATA_DIR}\reports\classification_report.csv"
test_img_path = rf"{DATA_DIR}\test5.JPG"
path2 = rf"{DATA_DIR}\opencv-greyscale.png"
model_path = r"C:\Rohan\Vector AI\repo\rohan_vai\src\models\trained_models\fmnist_model.h5"

class_names = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

# Data values
img_width = 28
img_height = 28

# CNN params
pool_size = 2
number_of_epochs = 50
number_of_iterations = 4
