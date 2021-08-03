# Build binary files of thirdparty tools used in GTSFM

# Assign a temporary directory for building binary files
TEMP_BUILD_DIR="./build"
mkdir -p $TEMP_BUILD_DIR
cd $TEMP_BUILD_DIR
echo "Temporary build path $TEMP_BUILD_DIR created."

ETH3D_MULTI_VIEW_EVALUATION_DIR="thirdparty/multi-view-evaluation"
ETH3D_MULTI_VIEW_EVALUATION_BINARY="ETH3DMultiViewEvaluation"
echo "Build $ETH3D_MULTI_VIEW_EVALUATION_DIR"

# Set C++ Flags
export CXXFLAGS="-std=c++11 -O3"
cmake ../$ETH3D_MULTI_VIEW_EVALUATION_DIR
make
cp ./$ETH3D_MULTI_VIEW_EVALUATION_BINARY ../$ETH3D_MULTI_VIEW_EVALUATION_DIR/$ETH3D_MULTI_VIEW_EVALUATION_BINARY
echo "Build $ETH3D_MULTI_VIEW_EVALUATION_DIR/$ETH3D_MULTI_VIEW_EVALUATION_BINARY successfully."

# Clean build path
cd ..
rm -rf $TEMP_BUILD_DIR
echo "Temporary build path $TEMP_BUILD_DIR cleaned."

