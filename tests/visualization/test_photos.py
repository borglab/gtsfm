"""Photo API tests: source association, bounded resize/cache, and path isolation."""

import json
from io import BytesIO
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient
from PIL import Image

from visualization.app import create_app
from visualization.photos import ScenePhotos


@pytest.fixture
def photo_scene(tmp_path: Path):
    results = tmp_path / "results"
    scene = results / "runs" / "brick" / "results" / "merged_retriangulated"
    scene.mkdir(parents=True)
    dataset = tmp_path / "source"
    (dataset / "image").mkdir(parents=True)
    Image.new("RGB", (1200, 800), (210, 40, 10)).save(dataset / "image" / "first photo.jpg")
    Image.new("RGB", (600, 900), (10, 150, 70)).save(dataset / "image" / "second.jpg")
    (scene / "images.txt").write_text(
        "# Two registered cameras, including an empty observation line\n"
        "8 1 0 0 0 0 0 0 0 first photo.jpg\n\n"
        "17 1 0 0 0 1 0 0 1 second.jpg\n0.5 0.2 -1\n"
    )
    (scene.parent / "config.yaml").write_text(
        yaml.safe_dump({"loader": {"_target_": "gtsfm.loader.Mobilebrick", "dataset_dir": str(dataset)}})
    )
    return results, scene, dataset


def test_saved_run_resolves_photos_without_live_job(photo_scene):
    results, scene, _ = photo_scene
    client = TestClient(create_app(results))
    response = client.get("/api/photos", params={"scene": scene.relative_to(results).as_posix()})
    assert response.status_code == 200
    items = response.json()["items"]
    assert [item["name"] for item in items] == ["first photo.jpg", "second.jpg"]
    image = client.get(items[0]["thumbnail"])
    assert image.status_code == 200
    assert image.headers["content-type"] == "image/jpeg"
    assert Image.open(BytesIO(image.content)).size == (256, 171)
    assert "private" in image.headers["cache-control"]
    assert client.get(items[0]["thumbnail"], headers={"If-None-Match": image.headers["etag"]}).status_code == 304
    assert Image.open(BytesIO(client.get(items[1]["preview"]).content)).size == (600, 900)


def test_retrieved_remote_run_uses_local_inputs_without_rewriting_evidence(photo_scene):
    results, scene, dataset = photo_scene
    config = scene.parent / "config.yaml"
    original = yaml.safe_dump({"loader": {"_target_": "Mobilebrick", "dataset_dir": "/remote/staged-inputs"}})
    config.write_text(original)
    (scene.parent.parent / "studio-inputs.json").write_text(json.dumps({"dataset_dir": str(dataset)}))
    manifest = ScenePhotos(results).manifest(scene.relative_to(results).as_posix())
    assert all(item["thumbnail"] for item in manifest["items"])
    assert config.read_text() == original


def test_placeholder_export_uses_recorded_loader_order(photo_scene):
    results, scene, dataset = photo_scene
    (scene / "images.txt").write_text("0 1 0 0 0 0 0 0 0 image_000000.jpg\n\n")
    (scene.parent.parent / "image_filenames.json").write_text(json.dumps(["first photo.jpg", "second.jpg"]))
    client = TestClient(create_app(results))
    manifest = client.get("/api/photos", params={"scene": scene.relative_to(results).as_posix()}).json()
    assert manifest["items"][0]["thumbnail"]
    assert client.get(manifest["items"][0]["thumbnail"]).status_code == 200


def test_actual_placeholder_like_filename_does_not_map_to_another_camera(photo_scene):
    results, scene, dataset = photo_scene
    (dataset / "image" / "second.jpg").rename(dataset / "image" / "image_000000.jpg")
    (scene / "images.txt").write_text("1 1 0 0 0 0 0 0 1 image_000000.jpg\n\n")
    (scene.parent.parent / "image_filenames.json").write_text(json.dumps(["first photo.jpg", "image_000000.jpg"]))
    store = ScenePhotos(results)
    assert store.image_source(scene.relative_to(results).as_posix(), 0).name == "image_000000.jpg"


def test_recorded_names_are_read_once_for_many_photos(photo_scene, monkeypatch):
    results, scene, _ = photo_scene
    mapping = scene.parent.parent / "image_filenames.json"
    mapping.write_text(json.dumps(["first photo.jpg", "second.jpg"]))
    read_text = Path.read_text
    reads = []

    def read(path, *args, **kwargs):
        if path == mapping:
            reads.append(path)
        return read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read)
    store = ScenePhotos(results)
    for _ in range(50):
        assert store.source_name(scene, 0, "image_000000.jpg") == "first photo.jpg"
    assert reads == [mapping]


def test_explicit_image_directory_and_missing_photo(photo_scene):
    results, scene, dataset = photo_scene
    (scene.parent / "config.yaml").write_text(
        yaml.safe_dump({"loader": {"images_dir": "image", "dataset_dir": str(dataset)}})
    )
    (dataset / "image" / "second.jpg").unlink()
    items = ScenePhotos(results).manifest(scene.relative_to(results).as_posix())["items"]
    assert items[0]["thumbnail"]
    assert items[1]["thumbnail"] is None


def test_photo_paths_cannot_escape_declared_dataset(photo_scene, tmp_path):
    results, scene, dataset = photo_scene
    external = tmp_path / "private.jpg"
    Image.new("RGB", (10, 10)).save(external)
    (dataset / "image" / "first photo.jpg").unlink()
    (dataset / "image" / "first photo.jpg").symlink_to(external)
    client = TestClient(create_app(results))
    relative = scene.relative_to(results).as_posix()
    assert client.get("/api/photos/image", params={"scene": relative, "index": 0}).status_code == 404
    assert client.get("/api/photos", params={"scene": "../source"}).status_code == 400
    assert ScenePhotos.source([dataset], "../private.jpg") is None
    assert ScenePhotos.source([dataset], str(external)) is None


@pytest.mark.parametrize("params", [{"size": 999999}, {"index": -1}, {"index": 99}])
def test_unknown_indices_and_unbounded_resize_are_rejected(photo_scene, params):
    results, scene, _ = photo_scene
    query = {"scene": scene.relative_to(results).as_posix(), "index": 0, **params}
    response = TestClient(create_app(results)).get("/api/photos/image", params=query)
    assert response.status_code in {404, 422}


def test_cache_is_bounded_and_source_changes_invalidate_it(photo_scene):
    results, scene, dataset = photo_scene
    store = ScenePhotos(results, cache_bytes=1500)
    first = dataset / "image" / "first photo.jpg"
    etag = store.etag(first, 256)
    original = store.resized(first, 256, etag)
    Image.new("RGB", (1200, 800), "blue").save(first)
    updated = store.etag(first, 256)
    assert updated != etag
    assert store.resized(first, 256, updated) != original
    second = dataset / "image" / "second.jpg"
    store.resized(second, 256, store.etag(second, 256))
    assert store._bytes <= store.cache_bytes


def test_exif_orientation_and_corrupt_images(photo_scene):
    results, scene, dataset = photo_scene
    exif = Image.Exif()
    exif[274] = 6
    Image.new("RGB", (1200, 800)).save(dataset / "image" / "first photo.jpg", exif=exif)
    client = TestClient(create_app(results))
    params = {"scene": scene.relative_to(results).as_posix(), "index": 0}
    image = client.get("/api/photos/image", params=params)
    assert Image.open(BytesIO(image.content)).size == (171, 256)
    (dataset / "image" / "second.jpg").write_bytes(b"not a photo")
    assert client.get("/api/photos/image", params={**params, "index": 1}).status_code == 400


def test_portable_export_without_config(photo_scene):
    results, scene, dataset = photo_scene
    (scene.parent / "config.yaml").unlink()
    (scene / "images").mkdir()
    (scene / "images" / "first photo.jpg").write_bytes((dataset / "image" / "first photo.jpg").read_bytes())
    items = ScenePhotos(results).manifest(scene.relative_to(results).as_posix())["items"]
    assert items[0]["thumbnail"]
    assert items[1]["thumbnail"] is None


def test_older_cli_relative_dataset_path(photo_scene, monkeypatch):
    results, scene, dataset = photo_scene
    monkeypatch.chdir(dataset.parent)
    (scene.parent / "config.yaml").write_text("loader:\n  _target_: gtsfm.loader.Mobilebrick\n  dataset_dir: source\n")
    assert ScenePhotos(results).manifest(scene.relative_to(results).as_posix())["items"][0]["thumbnail"]


def test_invalid_saved_config_is_reported_without_crashing(photo_scene):
    results, scene, _ = photo_scene
    (scene.parent / "config.yaml").write_text("loader: [invalid")
    response = TestClient(create_app(results)).get(
        "/api/photos", params={"scene": scene.relative_to(results).as_posix()}
    )
    assert response.status_code == 400


def test_extensionless_yfcc_identifiers(photo_scene):
    _, _, dataset = photo_scene
    assert ScenePhotos.source([dataset / "image"], "second") == dataset / "image" / "second.jpg"
