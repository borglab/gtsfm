#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 DOCKERHUB_NAMESPACE/gtsfm-modal-runtime [TAG]" >&2
  exit 2
fi

repository="${1#docker.io/}"
tag="${2:-firstclass}"
image="docker.io/${repository}:${tag}"

docker info >/dev/null
docker buildx build \
  --platform linux/amd64 \
  --file docker/modal-runtime.Dockerfile \
  --tag "${image}" \
  --output type=registry,compression=estargz,force-compression=true,oci-mediatypes=true \
  .

echo "Published ${image}"
