
# Download model weights for different modules available for use in GTSFM
# Note: SuperPoint and SuperGlue code and checkpoints may *not* be used for commercial purposes

#################### SuperPoint & SuperGlue ##########################

SUPERPOINT_CKPT_URL="https://github.com/magicleap/SuperGluePretrainedNetwork/raw/master/models/weights/superpoint_v1.pth"

SUPERGLUE_INDOOR_CKPT_URL="https://github.com/magicleap/SuperGluePretrainedNetwork/raw/master/models/weights/superglue_indoor.pth"
SUPERGLUE_OUTDOOR_CKPT_URL="https://github.com/magicleap/SuperGluePretrainedNetwork/raw/master/models/weights/superglue_outdoor.pth"

SUPERGLUE_WEIGHTS_DIR="./thirdparty/SuperGluePretrainedNetwork/models/weights"
SUPERPOINT_WEIGHTS_DIR=$SUPERGLUE_WEIGHTS_DIR

echo "Creating $SUPERGLUE_WEIGHTS_DIR"
mkdir -p $SUPERGLUE_WEIGHTS_DIR
mkdir -p $SUPERPOINT_WEIGHTS_DIR

wget -c --no-check-certificate -O $SUPERPOINT_WEIGHTS_DIR/superpoint_v1.pth $SUPERPOINT_CKPT_URL
wget -c --no-check-certificate -O $SUPERGLUE_WEIGHTS_DIR/superglue_indoor.pth $SUPERGLUE_INDOOR_CKPT_URL
wget -c --no-check-certificate -O $SUPERGLUE_WEIGHTS_DIR/superglue_outdoor.pth $SUPERGLUE_OUTDOOR_CKPT_URL

##################### PatchMatchNet ##########################

PATCHMATCHNET_WEIGHTS_DIR="./thirdparty/patchmatchnet/checkpoints"

echo $PATCHMATCHNET_WEIGHTS_DIR

echo "Creating $PATCHMATCHNET_WEIGHTS_DIR"
mkdir -p $PATCHMATCHNET_WEIGHTS_DIR

PATCHMATCHNET_URL="https://github.com/FangjinhuaWang/PatchmatchNet/raw/fa4ecae69b3a376ce238002db8d5283406128eac/checkpoints/model_000007.ckpt"

wget -c --no-check-certificate -O $PATCHMATCHNET_WEIGHTS_DIR/model_000007.ckpt $PATCHMATCHNET_URL

##################### D2Net #############################

D2NET_CKPT_URL="https://dusmanu.com/files/d2-net/d2_tf.pth"
D2NET_WEIGHTS_DIR="./thirdparty/d2net/weights"
mkdir -p $D2NET_WEIGHTS_DIR

wget $D2NET_CKPT_URL -O $D2NET_WEIGHTS_DIR/d2_tf.pth

##################### NetVLAD #############################

NETVLAD_WEIGHTS_DIR="./thirdparty/hloc/weights"
NETVLAD_CKPT_URL="https://github.com/johnwlambert/gtsfm-datasets-mirror/releases/download/gerrard-hall-100/VGG16-NetVLAD-Pitts30K.mat"
mkdir $NETVLAD_WEIGHTS_DIR
wget $NETVLAD_CKPT_URL -O $NETVLAD_WEIGHTS_DIR/VGG16-NetVLAD-Pitts30K.mat

##################### vggt #############################
VGGT_WEIGHTS_DIR="./thirdparty/vggt/weights"
VGGT_CKPT_URL="https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
mkdir $VGGT_WEIGHTS_DIR
wget $VGGT_CKPT_URL -O $VGGT_WEIGHTS_DIR/model.pt
