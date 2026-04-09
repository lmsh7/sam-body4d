# Copyright (c) Meta Platforms, Inc. and affiliates.
# PyTorch3D batch renderer — drop-in replacement for pyrender-based Renderer.

import math
import time
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    PerspectiveCameras,
    RasterizationSettings,
    MeshRasterizer,
    HardPhongShader,
    MeshRenderer,
    DirectionalLights,
    TexturesVertex,
    BlendParams,
)
from pytorch3d.renderer.lighting import DirectionalLights as _DLBase


def _raymond_light_directions() -> torch.Tensor:
    """Compute the 3 raymond directional-light directions.

    In pyrender each light node has a 4x4 matrix whose z-column is
    ``[xp, yp, zp]`` and the light shines along its **local -Z**,
    which in world space is ``-z_column``.  PyTorch3D's
    ``DirectionalLights`` expects the direction *toward* the light
    (i.e. from surface toward light), which equals ``+z_column``.
    """
    thetas = math.pi * np.array([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0])
    phis = math.pi * np.array([0.0, 2.0 / 3.0, 4.0 / 3.0])
    directions = []
    for phi, theta in zip(phis, thetas):
        xp = math.sin(theta) * math.cos(phi)
        yp = math.sin(theta) * math.sin(phi)
        zp = math.cos(theta)
        directions.append([xp, yp, zp])
    return torch.tensor(directions, dtype=torch.float32)  # (3, 3)


class MultiDirectionalLights(_DLBase):
    """DirectionalLights that sums contributions from *N* directions.

    PyTorch3D's built-in ``DirectionalLights`` only supports a single
    direction per batch element.  This subclass stores *N* directions
    and overrides ``diffuse()`` / ``specular()`` to accumulate all of
    them, matching the multi-light setup in pyrender.
    """

    def __init__(
        self,
        directions: torch.Tensor,          # (N, 3) — N light directions
        ambient_color: Tuple = (0.3, 0.3, 0.3),
        diffuse_color: Tuple = (1.0, 1.0, 1.0),
        specular_color: Tuple = (0.0, 0.0, 0.0),
        device: torch.device = None,
    ):
        # Initialize the parent with a dummy single direction.
        # We override diffuse/specular, so the parent's direction is unused.
        super().__init__(
            ambient_color=(ambient_color,),
            diffuse_color=(diffuse_color,),
            specular_color=(specular_color,),
            direction=((0.0, 0.0, 1.0),),
            device=device or directions.device,
        )
        # Store all N directions: (N, 3)
        self._multi_directions = directions.to(self.device)

    def diffuse(self, normals, points=None) -> torch.Tensor:
        """Sum diffuse from all N directional lights.

        normals: (B, ..., 3)
        returns: (B, ..., 3) diffuse color contribution
        """
        # diffuse_color from parent: (1, 3) or (B, 3)
        color = self.diffuse_color  # (1, 3)
        total = torch.zeros_like(normals[..., :3])

        for i in range(self._multi_directions.shape[0]):
            # direction toward light: (3,)
            d = self._multi_directions[i]
            # d dot normals => (B, ..., 1)
            cos_angle = (normals * d).sum(dim=-1, keepdim=True).clamp(min=0.0)
            total = total + color * cos_angle

        return total

    def specular(self, normals, points, camera_position, shininess) -> torch.Tensor:
        # No specular in our raymond lights setup
        return torch.zeros_like(normals[..., :3])


class PyTorch3DBatchRenderer:
    """Persistent GPU mesh renderer using PyTorch3D.

    Created **once** and reused across all batches/frames — unlike the
    pyrender ``Renderer`` which creates / destroys an ``OffscreenRenderer``
    on every single call.
    """

    # Upper bound on meshes rendered in a single GPU pass to cap VRAM.
    DEFAULT_SUB_BATCH = 64

    def __init__(self, device: torch.device = None):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        # Raymond directional lights (3 directions, white, intensity 1.0)
        self._light_directions = _raymond_light_directions().to(device)  # (3, 3)

        # Cache rasterizer settings keyed by (H, W)
        self._raster_cache: dict[Tuple[int, int], RasterizationSettings] = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_raster_settings(self, H: int, W: int) -> RasterizationSettings:
        key = (H, W)
        if key not in self._raster_cache:
            self._raster_cache[key] = RasterizationSettings(
                image_size=(H, W),
                blur_radius=0.0,
                faces_per_pixel=1,
                # bin_size omitted → PyTorch3D auto-selects coarse-to-fine
            )
        return self._raster_cache[key]

    def _build_lights(self) -> MultiDirectionalLights:
        """Build multi-directional raymond lights.

        3 directional lights at elevation pi/6, azimuths 0/120/240 deg,
        white color intensity 1.0 each, ambient (0.3, 0.3, 0.3).
        """
        return MultiDirectionalLights(
            directions=self._light_directions,
            ambient_color=(0.3, 0.3, 0.3),
            diffuse_color=(1.0, 1.0, 1.0),
            specular_color=(0.0, 0.0, 0.0),
            device=self.device,
        )

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
            Per-mesh vertices in HMR world coordinates (no flip needed).
        faces_list : list of (F_i, 3) int64 tensors on *device*
        colors_list : list of (V_i, 3) float32 RGB [0, 1] tensors on *device*
        focal_lengths : (B,) float32 tensor
        cam_translations : (B, 3) float32 tensor
            Raw ``cam_t`` from HMR (unmodified).
        image_size : (H, W) -- all images in this call share the same size.
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

        # --- Meshes ---
        _t0 = time.time()
        textures = TexturesVertex(verts_features=colors_list)
        meshes = Meshes(verts=verts_list, faces=faces_list, textures=textures)
        torch.cuda.synchronize(self.device)
        _t_meshes = time.time() - _t0

        # --- Cameras + Renderer setup ---
        _t0 = time.time()
        T = cam_translations.clone()

        R = torch.eye(3, device=self.device).unsqueeze(0).expand(B, -1, -1)
        fl = focal_lengths.unsqueeze(1).expand(-1, 2)  # (B, 2)  fx == fy
        pp = torch.tensor(
            [[W / 2.0, H / 2.0]], device=self.device
        ).expand(B, -1)  # (B, 2)

        cameras = PerspectiveCameras(
            focal_length=fl,
            principal_point=pp,
            R=R,
            T=T,
            image_size=((H, W),),
            in_ndc=False,
            device=self.device,
        )

        raster_settings = self._get_raster_settings(H, W)
        lights = self._build_lights()

        rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)
        shader = HardPhongShader(
            device=self.device,
            cameras=cameras,
            lights=lights,
            blend_params=BlendParams(background_color=(1.0, 1.0, 1.0)),
        )
        _t_setup = time.time() - _t0

        # --- Rasterize ---
        _t0 = time.time()
        fragments = rasterizer(meshes)
        torch.cuda.synchronize(self.device)
        _t_raster = time.time() - _t0

        # --- Shade ---
        _t0 = time.time()
        rgba = shader(fragments, meshes)
        torch.cuda.synchronize(self.device)
        _t_shade = time.time() - _t0

        rgb = rgba[..., :3]
        alpha = rgba[..., 3:4]

        # --- Alpha composite ---
        _t0 = time.time()
        if bg_images is not None:
            out = rgb * alpha + bg_images * (1.0 - alpha)
        else:
            out = rgb * alpha + (1.0 - alpha)  # white bg
        out = out.clamp(0.0, 1.0)
        torch.cuda.synchronize(self.device)
        _t_composite = time.time() - _t0

        print(f"    [CHUNK B={B}] meshes={_t_meshes:.3f}s setup={_t_setup:.3f}s raster={_t_raster:.3f}s shade={_t_shade:.3f}s composite={_t_composite:.3f}s")
        return out
