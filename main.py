import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import uvicorn
from api import app

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)