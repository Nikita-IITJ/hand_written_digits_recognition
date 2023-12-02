# Variables
REGISTRY_NAME="m22aie244"
DEP_IMAGE_NAME="dependencyimage"
FINAL_IMAGE_NAME="finalimage"
TAG="latest"

# Authenticate with Azure (if required)
sudo az acr login --name $REGISTRY_NAME

# Build Dependency Image
sudo docker build -t $DEP_IMAGE_NAME -f DependencyDockerfile .
sudo docker tag $DEP_IMAGE_NAME $REGISTRY_NAME.azurecr.io/$DEP_IMAGE_NAME:$TAG

# Push Dependency Image
sudo docker push $REGISTRY_NAME.azurecr.io/$DEP_IMAGE_NAME:$TAG

# Build Final Image
sudo docker build -t $FINAL_IMAGE_NAME -f FinalDockerfile .
sudo docker tag $FINAL_IMAGE_NAME $REGISTRY_NAME.azurecr.io/$FINAL_IMAGE_NAME:$TAG

# Push Final Image
sudo docker push $REGISTRY_NAME.azurecr.io/$FINAL_IMAGE_NAME:$TAG 