# Copyright (c) Meta Platforms, Inc. and affiliates.
# PyTorch3D batch renderer — drop-in replacement for pyrender-based Renderer.

import math
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


class PyTorch3DBatchRenderer:
    """Persistent GPU mesh renderer using PyTorch3D.

    Created **once** and reused across all batches/frames — unlike the
    pyrender ``Renderer`` which creates / destroys an ``OffscreenRenderer``
    on every single call.
    """

    # Upper bound on meshes rendered in a single GPU pass to cap VRAM.
    DEFAULT_SUB_BATCH = 16

    def __init__(self, device: torch.device = None):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        # Raymond directional lights (3 directions, white, intensity 1.0)
        light_dirs = _raymond_light_directions().to(device)  # (3, 3)
        # DirectionalLights expects (1, N, 3) for N lights
        self._light_directions = light_dirs.unsqueeze(0)  # (1, 3, 3)

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
                bin_size=0,  # use naive rasterization for robustness
            )
        return self._raster_cache[key]

    def _build_lights(self, batch_size: int) -> DirectionalLights:
        """Build DirectionalLights for *batch_size* meshes.

        PyTorch3D's ``DirectionalLights`` sums contributions from all
        supplied direction vectors.  We pass 3 directions (raymond),
        each with diffuse=1 and no specular, plus ambient=(0.3, 0.3, 0.3)
        matching the pyrender scene.
        """
        # Expand directions to (B, 3, 3)
        dirs = self._light_directions.expand(batch_size, -1, -1)
        return DirectionalLights(
            ambient_color=((0.3, 0.3, 0.3),),
            diffuse_color=((1.0, 1.0, 1.0),),
            specular_color=((0.0, 0.0, 0.0),),
            direction=dirs,
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
            Per-mesh vertices.  Vertices should already include the
            180° X-flip (``y, z *= -1``) applied by the caller.
        faces_list : list of (F_i, 3) int64 tensors on *device*
        colors_list : list of (V_i, 3) float32 RGB [0, 1] tensors on *device*
        focal_lengths : (B,) float32 tensor
        cam_translations : (B, 3) float32 tensor
            Same convention as pyrender: ``cam_t`` with ``[0] *= -1``
            already applied by caller.
        image_size : (H, W) — all images in this call share the same size.
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
        textures = TexturesVertex(verts_features=colors_list)
        meshes = Meshes(verts=verts_list, faces=faces_list, textures=textures)

        # --- Cameras ---
        # PyTorch3D screen-space convention (in_ndc=False):
        #   focal_length sign: positive = standard pinhole
        #   principal_point: (px, py) in pixels from top-left
        #   R: identity (no rotation)
        #   T: camera translation
        #
        # The pyrender code does:
        #   camera_translation[0] *= -1   (caller already did this)
        #   mesh gets 180° X-flip         (caller already did this on verts)
        #
        # In PyTorch3D screen coords the x-axis points right and y-axis
        # points down, which matches OpenCV / pyrender after the X-flip.
        # We need to negate T_y because PyTorch3D's T convention is
        # "translate the *world* relative to the camera", and its y-axis
        # in screen coords points down.

        T = cam_translations.clone()
        # T[:, 0] already negated by caller (matches pyrender)
        # T[:, 1] needs negation for PyTorch3D screen-coord convention
        T[:, 1] *= -1.0

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

        # --- Rasterizer + Shader ---
        raster_settings = self._get_raster_settings(H, W)
        lights = self._build_lights(B)

        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
            shader=HardPhongShader(
                device=self.device,
                cameras=cameras,
                lights=lights,
                blend_params=BlendParams(background_color=(1.0, 1.0, 1.0)),
            ),
        )

        # --- Render ---
        # Output: (B, H, W, 4) RGBA float [0, 1]
        rgba = renderer(meshes)
        rgb = rgba[..., :3]
        alpha = rgba[..., 3:4]

        # --- Alpha composite ---
        if bg_images is not None:
            out = rgb * alpha + bg_images * (1.0 - alpha)
        else:
            out = rgb * alpha + (1.0 - alpha)  # white bg

        return out.clamp(0.0, 1.0)
