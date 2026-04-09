"""3D Avatar loader — loads GLB/GLTF/VRM models and extracts blend shapes (morph targets).

Supports:
- GLTF 2.0 / GLB (standard 3D web format)
- VRM (VTuber standard, extends GLTF with blend shape groups)
- Avaturn avatars (GLB with ARKit blend shapes)
"""

from __future__ import annotations

import json
import logging
import struct
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class MeshData:
    """Parsed mesh data from a 3D model."""

    def __init__(self):
        self.vertices: Optional[np.ndarray] = None  # (N, 3) positions
        self.normals: Optional[np.ndarray] = None  # (N, 3)
        self.uvs: Optional[np.ndarray] = None  # (N, 2)
        self.indices: Optional[np.ndarray] = None  # (M,) triangle indices
        self.name: str = ""


class BlendShapeTarget:
    """A single blend shape / morph target."""

    def __init__(self, name: str):
        self.name = name
        self.position_deltas: Optional[np.ndarray] = None  # (N, 3)
        self.normal_deltas: Optional[np.ndarray] = None  # (N, 3)


class AvatarModel:
    """Loaded 3D avatar with meshes and blend shapes."""

    def __init__(self):
        self.name: str = ""
        self.meshes: list[MeshData] = []
        self.blend_shapes: dict[str, BlendShapeTarget] = {}
        self.blend_shape_names: list[str] = []
        self.skeleton: Optional[dict] = None
        self.textures: dict[str, np.ndarray] = {}
        # Raw GLTF data for renderer
        self.gltf_json: Optional[dict] = None
        self.gltf_bin: Optional[bytes] = None
        self.source_path: str = ""

    @property
    def has_blend_shapes(self) -> bool:
        return len(self.blend_shapes) > 0

    @property
    def vertex_count(self) -> int:
        return sum(m.vertices.shape[0] for m in self.meshes if m.vertices is not None)

    @property
    def face_count(self) -> int:
        return sum(
            m.indices.shape[0] // 3 for m in self.meshes if m.indices is not None
        )


class AvatarLoader:
    """Loads GLB/GLTF/VRM files into AvatarModel."""

    def load(self, path: str | Path) -> AvatarModel:
        path = Path(path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Avatar model not found: {path}")

        suffix = path.suffix.lower()
        if suffix == ".glb":
            return self._load_glb(path)
        elif suffix == ".gltf":
            return self._load_gltf(path)
        elif suffix == ".vrm":
            return self._load_glb(path)  # VRM is GLB with extensions
        else:
            raise ValueError(f"Unsupported format: {suffix}. Use .glb, .gltf, or .vrm")

    def _load_glb(self, path: Path) -> AvatarModel:
        """Load a binary GLTF (GLB) file."""
        with open(path, "rb") as f:
            # GLB header
            magic = struct.unpack("<I", f.read(4))[0]
            if magic != 0x46546C67:  # 'glTF'
                raise ValueError("Not a valid GLB file")
            version = struct.unpack("<I", f.read(4))[0]
            total_length = struct.unpack("<I", f.read(4))[0]

            # Chunk 0: JSON
            json_length = struct.unpack("<I", f.read(4))[0]
            json_type = struct.unpack("<I", f.read(4))[0]
            json_data = f.read(json_length)
            gltf = json.loads(json_data.decode("utf-8"))

            # Chunk 1: Binary buffer
            bin_data = b""
            if f.tell() < total_length:
                bin_length = struct.unpack("<I", f.read(4))[0]
                bin_type = struct.unpack("<I", f.read(4))[0]
                bin_data = f.read(bin_length)

        model = AvatarModel()
        model.name = path.stem
        model.gltf_json = gltf
        model.gltf_bin = bin_data
        model.source_path = str(path)

        self._parse_meshes(gltf, bin_data, model)
        self._parse_blend_shapes(gltf, bin_data, model)

        # Check for VRM extensions
        if "extensions" in gltf and "VRM" in gltf["extensions"]:
            self._parse_vrm_blendshapes(gltf, model)

        logger.info(
            f"Loaded avatar '{model.name}': "
            f"{len(model.meshes)} meshes, "
            f"{model.vertex_count} vertices, "
            f"{model.face_count} faces, "
            f"{len(model.blend_shapes)} blend shapes"
        )
        return model

    def _load_gltf(self, path: Path) -> AvatarModel:
        """Load a text GLTF file (JSON + separate .bin)."""
        with open(path) as f:
            gltf = json.load(f)

        # Load binary buffer
        bin_data = b""
        if gltf.get("buffers"):
            bin_uri = gltf["buffers"][0].get("uri", "")
            if bin_uri and not bin_uri.startswith("data:"):
                bin_path = path.parent / bin_uri
                if bin_path.exists():
                    bin_data = bin_path.read_bytes()

        model = AvatarModel()
        model.name = path.stem
        model.gltf_json = gltf
        model.gltf_bin = bin_data
        model.source_path = str(path)

        self._parse_meshes(gltf, bin_data, model)
        self._parse_blend_shapes(gltf, bin_data, model)

        logger.info(
            f"Loaded avatar '{model.name}': "
            f"{len(model.meshes)} meshes, "
            f"{len(model.blend_shapes)} blend shapes"
        )
        return model

    def _read_accessor(
        self, gltf: dict, bin_data: bytes, accessor_idx: int
    ) -> np.ndarray:
        """Read data from a GLTF accessor."""
        accessor = gltf["accessors"][accessor_idx]
        buffer_view = gltf["bufferViews"][accessor["bufferView"]]

        offset = buffer_view.get("byteOffset", 0) + accessor.get("byteOffset", 0)
        count = accessor["count"]

        # Component type
        comp_types = {5120: np.int8, 5121: np.uint8, 5122: np.int16,
                      5123: np.uint16, 5125: np.uint32, 5126: np.float32}
        dtype = comp_types[accessor["componentType"]]

        # Element size
        type_sizes = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}
        elem_size = type_sizes[accessor["type"]]

        total = count * elem_size
        data = np.frombuffer(bin_data, dtype=dtype, count=total, offset=offset)

        if elem_size > 1:
            data = data.reshape(count, elem_size)
        return data

    def _parse_meshes(self, gltf: dict, bin_data: bytes, model: AvatarModel) -> None:
        """Parse mesh primitives from GLTF."""
        if not bin_data:
            return

        for mesh_def in gltf.get("meshes", []):
            for prim in mesh_def.get("primitives", []):
                mesh = MeshData()
                mesh.name = mesh_def.get("name", "unnamed")
                attrs = prim.get("attributes", {})

                if "POSITION" in attrs:
                    mesh.vertices = self._read_accessor(gltf, bin_data, attrs["POSITION"])
                if "NORMAL" in attrs:
                    mesh.normals = self._read_accessor(gltf, bin_data, attrs["NORMAL"])
                if "TEXCOORD_0" in attrs:
                    mesh.uvs = self._read_accessor(gltf, bin_data, attrs["TEXCOORD_0"])
                if "indices" in prim:
                    mesh.indices = self._read_accessor(
                        gltf, bin_data, prim["indices"]
                    ).flatten()

                model.meshes.append(mesh)

    def _parse_blend_shapes(
        self, gltf: dict, bin_data: bytes, model: AvatarModel
    ) -> None:
        """Parse morph targets (blend shapes) from GLTF meshes."""
        if not bin_data:
            return

        for mesh_def in gltf.get("meshes", []):
            # Blend shape names from extras or targetNames
            target_names = mesh_def.get("extras", {}).get("targetNames", [])

            for prim in mesh_def.get("primitives", []):
                targets = prim.get("targets", [])
                for i, target in enumerate(targets):
                    name = target_names[i] if i < len(target_names) else f"morph_{i}"

                    bs = BlendShapeTarget(name)
                    if "POSITION" in target:
                        bs.position_deltas = self._read_accessor(
                            gltf, bin_data, target["POSITION"]
                        )
                    if "NORMAL" in target:
                        bs.normal_deltas = self._read_accessor(
                            gltf, bin_data, target["NORMAL"]
                        )

                    model.blend_shapes[name] = bs
                    if name not in model.blend_shape_names:
                        model.blend_shape_names.append(name)

    def _parse_vrm_blendshapes(self, gltf: dict, model: AvatarModel) -> None:
        """Parse VRM-specific blend shape groups (maps VRM names to morph targets)."""
        vrm = gltf["extensions"]["VRM"]
        groups = vrm.get("blendShapeMaster", {}).get("blendShapeGroups", [])

        for group in groups:
            preset = group.get("presetName", group.get("name", ""))
            # VRM presets map to our blend shape names
            # (e.g., "a" → mouth open, "i" → ee, "u" → oo, etc.)
            if preset:
                logger.debug(f"VRM blend shape group: {preset}")
