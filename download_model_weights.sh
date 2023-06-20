
# Download model weights for different modules available for use in GTSFM
# Note: SuperPoint and SuperGlue code and checkpoints may *not* be used for commercial purposes

##################### SuperPoint & SuperGlue ##########################

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

D2NET_CKPT_URL="https://dsmn.ml/files/d2-net/d2_tf.pth"
D2NET_WEIGHTS_DIR="./thirdparty/d2net/weights"
mkdir -p $D2NET_WEIGHTS_DIR

wget $D2NET_CKPT_URL -O $D2NET_WEIGHTS_DIR/d2_tf.pth

##################### NetVLAD #############################

python thirdparty/hloc/download_netvlad.py