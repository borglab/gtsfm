"""Script for downoading the NetVLAD weights.

Note: Copied from NetVLAD.__init__.
"""

import logging
import subprocess
from pathlib import Path


logger = logging.getLogger(__name__)

# path to /thirdparty/hloc/weights/{CHECKPOINT}.mat
netvlad_path = Path(__file__).resolve().parent / "weights"

conf = {"model_name": "VGG16-NetVLAD-Pitts30K", "checkpoint_dir": netvlad_path, "whiten": True}

dir_models = {
    "VGG16-NetVLAD-Pitts30K": "https://cvg-data.inf.ethz.ch/hloc/netvlad/Pitts30K_struct.mat",
    "VGG16-NetVLAD-TokyoTM": "https://cvg-data.inf.ethz.ch/hloc/netvlad/TokyoTM_struct.mat",
}

# Download the checkpoint.
checkpoint = conf["checkpoint_dir"] / str(conf["model_name"] + ".mat")
if not checkpoint.exists():
    checkpoint.parent.mkdir(exist_ok=True)
    link = dir_models[conf["model_name"]]
    cmd = ["wget", link, "-O", str(checkpoint)]
    logger.info(f"Downloading the NetVLAD model with `{cmd}`.")
    subprocess.run(cmd, check=True)
