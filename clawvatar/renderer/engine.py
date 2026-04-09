"""3D rendering engine using moderngl — offscreen headless rendering.

Renders 3D avatar meshes with blend shape deformations. Supports:
- EGL headless context (Linux, no display needed)
- osmesa fallback (software rendering)
"""

from __future__ import annotations

import logging
from typing import Optional

import moderngl
import numpy as np

logger = logging.getLogger(__name__)

# Simplified vertex shader — blend shapes applied on CPU for flexibility
VERTEX_SHADER = """
#version 330

uniform mat4 u_projection;
uniform mat4 u_view;
uniform mat4 u_model;
uniform vec3 u_light_dir;

in vec3 in_position;
in vec3 in_normal;
in vec2 in_uv;

out vec3 v_normal;
out vec2 v_uv;
out float v_diffuse;

void main() {
    vec4 world_pos = u_model * vec4(in_position, 1.0);
    gl_Position = u_projection * u_view * world_pos;

    v_normal = normalize(mat3(u_model) * in_normal);
    v_uv = in_uv;
    v_diffuse = max(dot(v_normal, normalize(u_light_dir)), 0.0);
}
"""

FRAGMENT_SHADER = """
#version 330

uniform vec3 u_base_color;
uniform float u_ambient;

in vec3 v_normal;
in vec2 v_uv;
in float v_diffuse;

out vec4 fragColor;

void main() {
    float light = u_ambient + (1.0 - u_ambient) * v_diffuse;
    // Use UV to slightly modulate color (prevents attribute from being optimized out)
    vec3 color = u_base_color + vec3(v_uv * 0.001, 0.0);
    fragColor = vec4(color * light, 1.0);
}
"""


def _perspective(fov_deg: float, aspect: float, near: float, far: float) -> np.ndarray:
    f = 1.0 / np.tan(np.radians(fov_deg) / 2.0)
    m = np.zeros((4, 4), dtype=np.float32)
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = (2.0 * far * near) / (near - far)
    m[3, 2] = -1.0
    return m


def _look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    f = target - eye
    f = f / np.linalg.norm(f)
    s = np.cross(f, up)
    s = s / np.linalg.norm(s)
    u = np.cross(s, f)

    m = np.eye(4, dtype=np.float32)
    m[0, :3] = s
    m[1, :3] = u
    m[2, :3] = -f
    m[0, 3] = -np.dot(s, eye)
    m[1, 3] = -np.dot(u, eye)
    m[2, 3] = np.dot(f, eye)
    return m


def _rotation_matrix(yaw: float, pitch: float, roll: float) -> np.ndarray:
    yaw, pitch, roll = np.radians([yaw, pitch, roll])
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)

    m = np.eye(4, dtype=np.float32)
    m[0, 0] = cy * cr + sy * sp * sr
    m[0, 1] = -cy * sr + sy * sp * cr
    m[0, 2] = sy * cp
    m[1, 0] = sr * cp
    m[1, 1] = cr * cp
    m[1, 2] = -sp
    m[2, 0] = -sy * cr + cy * sp * sr
    m[2, 1] = sy * sr + cy * sp * cr
    m[2, 2] = cy * cp
    return m


class MeshRenderData:
    """GPU data for a single mesh."""

    def __init__(self):
        self.vao: Optional[moderngl.VertexArray] = None
        self.vbo: Optional[moderngl.Buffer] = None
        self.ibo: Optional[moderngl.Buffer] = None
        self.index_count: int = 0
        # CPU-side data for blend shape deformation
        self.base_vertices: Optional[np.ndarray] = None
        self.base_normals: Optional[np.ndarray] = None
        self.uvs: Optional[np.ndarray] = None
        self.indices: Optional[np.ndarray] = None
        # Blend shape deltas: name -> (N, 3) position deltas
        self.blend_deltas: dict[str, np.ndarray] = {}


class Renderer3D:
    """Offscreen 3D renderer using moderngl."""

    def __init__(self, width: int = 512, height: int = 512):
        self.width = width
        self.height = height
        self.ctx: Optional[moderngl.Context] = None
        self.prog: Optional[moderngl.Program] = None
        self.fbo: Optional[moderngl.Framebuffer] = None
        self._meshes: list[MeshRenderData] = []

        # Camera
        self.camera_distance = 0.6
        self.camera_fov = 30.0
        self.camera_target = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        # Lighting
        self.light_dir = np.array([0.3, 0.8, 0.5], dtype=np.float32)
        self.ambient = 0.35
        self.base_color = np.array([0.85, 0.75, 0.7], dtype=np.float32)

        # Background
        self.bg_color = (0.15, 0.15, 0.18, 1.0)

    def initialize(self) -> None:
        """Create OpenGL context (headless)."""
        try:
            self.ctx = moderngl.create_standalone_context(backend="egl")
            logger.info("Created EGL headless context")
        except Exception:
            try:
                self.ctx = moderngl.create_standalone_context()
                logger.info("Created standalone context (osmesa/display)")
            except Exception as e:
                raise RuntimeError(
                    f"Failed to create OpenGL context: {e}. "
                    "Install libegl1-mesa-dev or libosmesa6-dev."
                )

        self.prog = self.ctx.program(
            vertex_shader=VERTEX_SHADER,
            fragment_shader=FRAGMENT_SHADER,
        )

        # Create framebuffer
        color = self.ctx.texture((self.width, self.height), 4)
        depth = self.ctx.depth_renderbuffer((self.width, self.height))
        self.fbo = self.ctx.framebuffer(color_attachments=[color], depth_attachment=depth)

        self.ctx.enable(moderngl.DEPTH_TEST)
        # Don't cull faces — VRM models often have mixed winding
        self.ctx.disable(moderngl.CULL_FACE)

        logger.info(f"Renderer initialized: {self.width}x{self.height}")

    def load_avatar(self, avatar) -> None:
        """Load avatar meshes into GPU buffers."""
        from clawvatar.avatar.loader import AvatarModel

        self._meshes.clear()

        for mesh in avatar.meshes:
            if mesh.vertices is None or mesh.indices is None:
                continue

            rd = MeshRenderData()
            rd.base_vertices = mesh.vertices.astype(np.float32).copy()
            rd.base_normals = (
                mesh.normals.astype(np.float32).copy()
                if mesh.normals is not None
                else np.zeros_like(rd.base_vertices)
            )
            rd.uvs = (
                mesh.uvs.astype(np.float32).copy()
                if mesh.uvs is not None
                else np.zeros((len(rd.base_vertices), 2), dtype=np.float32)
            )
            rd.indices = mesh.indices.astype(np.uint32).copy()
            rd.index_count = len(rd.indices)

            # Collect blend shape deltas for this mesh
            for bs_name, bs in avatar.blend_shapes.items():
                if bs.position_deltas is not None and len(bs.position_deltas) == len(rd.base_vertices):
                    rd.blend_deltas[bs_name] = bs.position_deltas.astype(np.float32)

            # Build initial GPU buffers
            self._upload_mesh(rd)
            self._meshes.append(rd)

        # Auto-center camera
        if avatar.meshes and avatar.meshes[0].vertices is not None:
            all_verts = np.vstack(
                [m.vertices for m in avatar.meshes if m.vertices is not None]
            )
            center = all_verts.mean(axis=0)
            extent = all_verts.max(axis=0) - all_verts.min(axis=0)
            self.camera_target = center.astype(np.float32)
            self.camera_distance = float(np.max(extent)) * 1.2

        logger.info(
            f"Avatar loaded into renderer: {len(self._meshes)} meshes, "
            f"blend shapes per mesh: {[len(m.blend_deltas) for m in self._meshes]}"
        )

    def _upload_mesh(self, rd: MeshRenderData) -> None:
        """Upload/re-upload mesh data to GPU."""
        vertex_data = np.hstack([rd.base_vertices, rd.base_normals, rd.uvs]).astype(np.float32)

        if rd.vbo is not None:
            rd.vbo.write(vertex_data.tobytes())
        else:
            rd.vbo = self.ctx.buffer(vertex_data.tobytes())
            rd.ibo = self.ctx.buffer(rd.indices.tobytes())
            rd.vao = self.ctx.vertex_array(
                self.prog,
                [(rd.vbo, "3f 3f 2f", "in_position", "in_normal", "in_uv")],
                index_buffer=rd.ibo,
            )

    def _apply_blend_shapes(self, rd: MeshRenderData, weights: dict[str, float]) -> None:
        """Apply blend shape deformations on CPU and re-upload vertices."""
        deformed = rd.base_vertices.copy()
        for name, w in weights.items():
            if w > 0.001 and name in rd.blend_deltas:
                deformed += rd.blend_deltas[name] * w

        # Rebuild vertex buffer with deformed positions
        vertex_data = np.hstack([deformed, rd.base_normals, rd.uvs]).astype(np.float32)
        rd.vbo.write(vertex_data.tobytes())

    def render(
        self,
        blend_weights: dict[str, float],
        head_yaw: float = 0.0,
        head_pitch: float = 0.0,
        head_roll: float = 0.0,
    ) -> np.ndarray:
        """Render one frame with given blend shape weights and head pose.

        Returns:
            BGR image (H, W, 3) as numpy array.
        """
        self.fbo.use()
        self.ctx.clear(*self.bg_color)

        # Camera — VRM models face -Z, so camera goes to -Z to see the front
        aspect = self.width / self.height
        proj = _perspective(self.camera_fov, aspect, 0.01, 100.0)
        eye = self.camera_target + np.array(
            [0.0, 0.0, -self.camera_distance], dtype=np.float32
        )
        view = _look_at(eye, self.camera_target, np.array([0, 1, 0], dtype=np.float32))
        model = _rotation_matrix(head_yaw, head_pitch, head_roll)

        # Set uniforms
        self.prog["u_projection"].write(proj.T.tobytes())
        self.prog["u_view"].write(view.T.tobytes())
        self.prog["u_model"].write(model.T.tobytes())
        self.prog["u_light_dir"].value = tuple(self.light_dir)
        self.prog["u_ambient"].value = self.ambient
        self.prog["u_base_color"].value = tuple(self.base_color)

        # Apply blend shapes and draw each mesh
        for rd in self._meshes:
            if blend_weights:
                self._apply_blend_shapes(rd, blend_weights)
            rd.vao.render(moderngl.TRIANGLES)

        # Read pixels
        data = self.fbo.color_attachments[0].read()
        image = np.frombuffer(data, dtype=np.uint8).reshape(self.height, self.width, 4)
        image = np.flipud(image)
        bgr = image[:, :, [2, 1, 0]]
        return bgr.copy()

    def cleanup(self) -> None:
        if self.ctx:
            self.ctx.release()
            self.ctx = None
