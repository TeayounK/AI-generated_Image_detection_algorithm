import os
import kagglehub

os.environ["KAGGLEHUB_CACHE"] = "./data/"
path = kagglehub.dataset_download("anhphmminh/cnnspot")
