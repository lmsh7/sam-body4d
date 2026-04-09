# nvdiffrast batch renderer — drop-in replacement for PyTorch3D-based Renderer.

import math
import time
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import nvdiffrast.torch as dr


def _raymond_light_directions() -> torch.Tensor:
    """Compute the 3 raymond directional-light directions (toward light)."""
    thetas = math.pi * np.array([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0])
    phis = math.pi * np.array([0.0, 2.0 / 3.0, 4.0 / 3.0])
    directions = []
    for phi, theta in zip(phis, thetas):
        xp = math.sin(theta) * math.cos(phi)
        yp = math.sin(theta) * math.sin(phi)
        zp = math.cos(theta)
        directions.append([xp, yp, zp])
    return torch.tensor(directions, dtype=torch.float32)  # (3, 3)


class NvDiffrastBatchRenderer:
    """Persistent GPU mesh renderer using nvdiffrast.

    Created **once** and reused across all batches/frames.
    Drop-in replacement for ``PyTorch3DBatchRenderer``.
    """

    DEFAULT_SUB_BATCH = 64

    def __init__(self, device: torch.device = None):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        self.glctx = dr.RasterizeCudaContext(device=device)

        self._light_directions = _raymond_light_directions().to(device)  # (3, 3)
        self._ambient = torch.tensor([0.3, 0.3, 0.3], dtype=torch.float32, device=device)
        self._diffuse = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32, device=device)

        self._near = 0.01
        self._far = 100.0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_projection(
        self, focal_lengths: torch.Tensor, H: int, W: int,
    ) -> torch.Tensor:
        """Build (B, 4, 4) projection matrices for nvdiffrast.

        The input vertices live in a coordinate system where the camera
        looks along **+Z** (HMR convention: cam_t.z is positive depth).
        Standard OpenGL looks along -Z, so we negate Z in the matrix to
        map positive-Z depth into the valid clip-space range.

        Concretely, for a camera-space point (X, Y, Z) with Z > 0:
          x_clip =  (2f/W) * X
          y_clip =  (2f/H) * Y
          z_clip =  (f+n)/(f-n) * Z  - 2fn/(f-n)   (maps [n,f] → [-1,1])
          w_clip =  Z                                (positive)

        After perspective divide, nvdiffrast rasterises in the cube
        [-1,1]^3.  We flip the rendered image vertically afterwards to
        go from OpenGL Y-up to screen Y-down.
        """
        B = focal_lengths.shape[0]
        n, f = self._near, self._far

        proj = torch.zeros(B, 4, 4, dtype=torch.float32, device=self.device)
        proj[:, 0, 0] = 2.0 * focal_lengths / W
        proj[:, 1, 1] = 2.0 * focal_lengths / H
        proj[:, 2, 2] = (f + n) / (f - n)
        proj[:, 2, 3] = -2.0 * f * n / (f - n)
        proj[:, 3, 2] = 1.0
        return proj

    def _compute_vertex_normals(
        self, verts: torch.Tensor, faces: torch.Tensor,
    ) -> torch.Tensor:
        """Compute area-weighted vertex normals.

        verts: (B, V, 3)   faces: (F, 3) int32
        Returns: (B, V, 3) normalized
        """
        B, V, _ = verts.shape
        v0 = verts[:, faces[:, 0]]  # (B, F, 3)
        v1 = verts[:, faces[:, 1]]
        v2 = verts[:, faces[:, 2]]

        face_normals = torch.cross(v1 - v0, v2 - v0, dim=-1)  # (B, F, 3)

        # Scatter-add face normals to vertices
        vertex_normals = torch.zeros_like(verts)  # (B, V, 3)
        F_count = faces.shape[0]
        for i in range(3):
            idx = faces[:, i].unsqueeze(0).unsqueeze(-1).expand(B, F_count, 3)
            vertex_normals.scatter_add_(1, idx, face_normals)

        return F.normalize(vertex_normals, dim=-1)

    def _phong_shade(
        self,
        colors: torch.Tensor,
        normals: torch.Tensor,
    ) -> torch.Tensor:
        """Apply raymond Phong shading (3 directional lights, no specular).

        colors:  (B, H, W, 3) interpolated vertex colors
        normals: (B, H, W, 3) interpolated vertex normals
        Returns: (B, H, W, 3) shaded RGB
        """
        normals = F.normalize(normals, dim=-1)

        diffuse_total = torch.zeros_like(colors)
        for i in range(self._light_directions.shape[0]):
            d = self._light_directions[i]  # (3,)
            cos_angle = (normals * d).sum(dim=-1, keepdim=True).clamp(min=0.0)
            diffuse_total = diffuse_total + self._diffuse * cos_angle

        shaded = colors * (self._ambient + diffuse_total)
        return shaded.clamp(0.0, 1.0)

    # ------------------------------------------------------------------
    # Rendering paths
    # ------------------------------------------------------------------

    def _render_uniform_batch(
        self,
        verts_list: List[torch.Tensor],
        faces: torch.Tensor,
        colors_list: List[torch.Tensor],
        focal_lengths: torch.Tensor,
        cam_translations: torch.Tensor,
        H: int, W: int,
        bg_images: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Render a batch where all meshes share the same topology (V, F)."""
        B = len(verts_list)

        verts = torch.stack(verts_list)    # (B, V, 3)
        colors = torch.stack(colors_list)  # (B, V, 3)

        # Camera-space position
        verts_cam = verts + cam_translations.unsqueeze(1)  # (B, V, 3)

        # Vertex normals
        normals = self._compute_vertex_normals(verts_cam, faces)  # (B, V, 3)

        # Projection to clip space
        proj = self._build_projection(focal_lengths, H, W)  # (B, 4, 4)
        ones = torch.ones(B, verts_cam.shape[1], 1, dtype=torch.float32, device=self.device)
        pos_homo = torch.cat([verts_cam, ones], dim=-1)      # (B, V, 4)
        pos_clip = torch.bmm(pos_homo, proj.transpose(1, 2))  # (B, V, 4)

        # Rasterize
        rast_out, _ = dr.rasterize(self.glctx, pos_clip, faces, resolution=[H, W])
        # rast_out: (B, H, W, 4)

        # Interpolate attributes
        interp_colors, _ = dr.interpolate(colors, rast_out, faces)    # (B, H, W, 3)
        interp_normals, _ = dr.interpolate(normals, rast_out, faces)  # (B, H, W, 3)

        # Mask: triangle_id > 0 means fragment is covered
        mask = (rast_out[..., 3:4] > 0).float()  # (B, H, W, 1)

        # Shade
        shaded = self._phong_shade(interp_colors, interp_normals)

        # Alpha composite
        if bg_images is not None:
            out = shaded * mask + bg_images * (1.0 - mask)
        else:
            out = shaded * mask + (1.0 - mask)  # white background

        # Flip Y: OpenGL Y-up → screen Y-down
        out = out.flip(1).clamp(0.0, 1.0)
        return out

    def _render_variable_batch(
        self,
        verts_list: List[torch.Tensor],
        faces_list: List[torch.Tensor],
        colors_list: List[torch.Tensor],
        focal_lengths: torch.Tensor,
        cam_translations: torch.Tensor,
        H: int, W: int,
        bg_images: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Render a batch where meshes have different topologies (per-element)."""
        results = []
        for i in range(len(verts_list)):
            out = self._render_uniform_batch(
                [verts_list[i]],
                faces_list[i][:, [0, 2, 1]].contiguous().to(torch.int32),
                [colors_list[i]],
                focal_lengths[i:i+1],
                cam_translations[i:i+1],
                H, W,
                bg_images[i:i+1] if bg_images is not None else None,
            )
            results.append(out)
        return torch.cat(results, dim=0)

    def _render_chunk(
        self,
        verts_list: List[torch.Tensor],
        faces_list: List[torch.Tensor],
        colors_list: List[torch.Tensor],
        focal_lengths: torch.Tensor,
        cam_translations: torch.Tensor,
        image_size: Tuple[int, int],
        bg_images: Optional[torch.Tensor],
    ) -> torch.Tensor:
        B = len(verts_list)
        H, W = image_size

        _t0 = time.time()

        # Check if all meshes share the same topology
        uniform = (
            all(v.shape[0] == verts_list[0].shape[0] for v in verts_list) and
            all(f.shape[0] == faces_list[0].shape[0] for f in faces_list)
        )

        if uniform:
            # Reverse winding order: the Y-flip in vis_utils.py turns CCW
            # faces into CW.  nvdiffrast culls CW (back-facing), so we swap
            # columns 1 & 2 to restore CCW front-face orientation.
            faces_i32 = faces_list[0][:, [0, 2, 1]].contiguous().to(torch.int32)
            out = self._render_uniform_batch(
                verts_list, faces_i32, colors_list,
                focal_lengths, cam_translations, H, W, bg_images,
            )
        else:
            out = self._render_variable_batch(
                verts_list, faces_list, colors_list,
                focal_lengths, cam_translations, H, W, bg_images,
            )

        torch.cuda.synchronize(self.device)
        _elapsed = time.time() - _t0
        print(f"    [nvdiffrast CHUNK B={B}] {_elapsed:.3f}s")
        return out

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def render_batch(
        self,
        verts_list: List[torch.Tensor],
        faces_list: List[torch.Tensor],
        colors_list: List[torch.Tensor],
        focal_lengths: torch.Tensor,
        cam_translations: torch.Tensor,
        image_size: Tuple[int, int],
        bg_images: Optional[torch.Tensor] = None,
        sub_batch_size: int = DEFAULT_SUB_BATCH,
    ) -> torch.Tensor:
        """Render *B* meshes in (possibly chunked) GPU passes.

        Parameters
        ----------
        verts_list : list of (V_i, 3) float32 tensors on *device*
        faces_list : list of (F_i, 3) int64 tensors on *device*
        colors_list : list of (V_i, 3) float32 RGB [0, 1] tensors on *device*
        focal_lengths : (B,) float32 tensor
        cam_translations : (B, 3) float32 tensor
        image_size : (H, W)
        bg_images : (B, H, W, 3) float32 [0, 1] or *None* for white bg.
        sub_batch_size : max meshes per GPU pass.

        Returns
        -------
        (B, H, W, 3) float32 tensor in [0, 1].
        """
        B = len(verts_list)
        H, W = image_size

        if B == 0:
            if bg_images is not None:
                return bg_images.clone()
            return torch.ones(0, H, W, 3, device=self.device)

        results = []
        for start in range(0, B, sub_batch_size):
            end = min(start + sub_batch_size, B)
            chunk = self._render_chunk(
                verts_list[start:end],
                faces_list[start:end],
                colors_list[start:end],
                focal_lengths[start:end],
                cam_translations[start:end],
                (H, W),
                bg_images[start:end] if bg_images is not None else None,
            )
            results.append(chunk)

        return torch.cat(results, dim=0)
