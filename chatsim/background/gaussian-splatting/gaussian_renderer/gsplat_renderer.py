import torch
from torch.nn import functional as F
from gsplat import rasterization
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
import omegaconf
import math
from utils.graphics_utils import OETF


def gsplat_render(viewpoint_camera, pc : GaussianModel, args: omegaconf.dictconfig.DictConfig, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, exposure_scale = None, debug=False):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    # Set up rasterization configuration
    if viewpoint_camera.K is not None:
        focal_length_x, focal_length_y, cx, cy = viewpoint_camera.K
        K = torch.tensor([
            [focal_length_x, 0, cx],
            [0, focal_length_y, cy],
            [0, 0, 1.0]
        ]).to(pc.get_xyz)
    else:
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
        focal_length_x = viewpoint_camera.image_width / (2 * tanfovx)
        focal_length_y = viewpoint_camera.image_height / (2 * tanfovy)
        K = torch.tensor(
            [
                [focal_length_x, 0, viewpoint_camera.image_width / 2.0],
                [0, focal_length_y, viewpoint_camera.image_height / 2.0],
                [0, 0, 1],
            ]
        ).to(pc.get_xyz)

    means3D = pc.get_xyz
    opacity = pc.get_opacity
    scales = pc.get_scaling * scaling_modifier
    rotations = pc.get_rotation

    if override_color is not None:
        colors = override_color # [N, 3]
        sh_degree = None
    else:
        colors = pc.get_features # [N, K, 3]
        sh_degree = pc.active_sh_degree

    viewmat = viewpoint_camera.world_view_transform.transpose(0, 1) # [4, 4]

    if debug:
        # --- START DEBUGGING BLOCK ---
        if torch.isnan(means3D).any() or torch.isinf(means3D).any():
            raise("!!! NaN or Inf detected in means3D !!!")
        if torch.isnan(rotations).any() or torch.isinf(rotations).any():
            raise("!!! NaN or Inf detected in rotations !!!")
        if torch.isnan(scales).any() or torch.isinf(scales).any():
            raise("!!! NaN or Inf detected in scales !!!")
        # Scales must be positive, this is a very important check!
        if (scales <= 0).any():
            raise(f"!!! Zero or negative value detected in scales. Min scale: {scales.min().item()} !!!")
        if torch.isnan(opacity).any() or torch.isinf(opacity).any():
            raise("!!! NaN or Inf detected in opacity !!!")
        if torch.isnan(colors).any() or torch.isinf(colors).any():
            raise("!!! NaN or Inf detected in colors !!!")
        # --- END DEBUGGING BLOCK ---

        # ---- Sanity + stabilization before rasterization ----
        def _finite(name, t):
            if not torch.is_tensor(t): return
            if not torch.all(torch.isfinite(t)):
                bad = torch.isnan(t).sum().item(), torch.isinf(t).sum().item()
                raise RuntimeError(f"[NaN/Inf] {name} has NaN/Inf: nan={bad[0]} inf={bad[1]}")

        # clamp / sanitize scales and opacities
        eps = 1e-6
        scales = torch.clamp(scales, min=eps)  # avoid σ=0

        # normalize quats (zero-out bad ones to identity)
        quat_norm = torch.linalg.norm(rotations, dim=-1, keepdim=True)
        bad_quat = quat_norm.squeeze(-1) < eps
        rotations = rotations / torch.clamp(quat_norm, min=eps)
        if bad_quat.any():
            print(f"Warning: {bad_quat.sum().item()} / {len(bad_quat)} bad quaternions found, replacing with identity.")
            # replace garbage with identity quaternion [1,0,0,0] (w,x,y,z)
            rotations[bad_quat] = rotations.new_tensor([1.0, 0.0, 0.0, 0.0])

        # keep opacities in (0,1)
        op = opacity.squeeze(-1)
        op = torch.clamp(op, 0.0, 1.0)

        # finiteness checks
        _finite("means3D", means3D)
        _finite("scales", scales)
        _finite("rotations", rotations)
        _finite("opacities", op)
        _finite("K", K)
        _finite("viewmat", viewmat)

        if not (K[0,0] > 0 and K[1,1] > 0):
            raise RuntimeError("Non-positive focal length (fx/fy)")

        # (optional) downweight insane scales to avoid huge exponents
        max_scale = 1e3
        if (scales > max_scale).any():
            scales = torch.clamp(scales, max=max_scale)

        if means3D.numel() == 0:
            raise RuntimeError("No Gaussians to render!")

    render_colors, render_alphas, info = rasterization(
        means=means3D,    # [N, 3]
        quats=rotations,  # [N, 4]
        scales=scales,    # [N, 3]
        opacities=opacity.squeeze(-1),  # [N,] (op)
        colors=colors,
        viewmats=viewmat[None],  # [1, 4, 4]
        Ks=K[None],  # [1, 3, 3]
        backgrounds=None,
        width=int(viewpoint_camera.image_width),
        height=int(viewpoint_camera.image_height),
        packed=False,
        sh_degree=sh_degree,
        render_mode='RGB+ED',
    )
    # [1, H, W, 4] -> [3, H, W]
    rendered_image = render_colors[0].permute(2, 0, 1)[:3]
    # [1, H, W, 4] -> [1, H, W]
    rendered_depth = render_colors[0].permute(2, 0, 1)[3:]
    # [1, H, W, 1] -> [1, H, W]
    rendered_alphas = render_alphas[0].permute(2, 0, 1)

    if exposure_scale is not None:
        rendered_image *= exposure_scale
        rendered_image = OETF(rendered_image)

    radii = info["radii"].squeeze(0) # [N,]
    if radii.dim() == 2:
        radii = radii.amax(dim=-1)
    try:
        info["means2d"].retain_grad() # [1, N, 2]
    except:
        pass

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return_pkg =  {"render_3dgs": rendered_image,
                    "viewspace_points": info["means2d"],
                    "visibility_filter" : radii > 0,
                    "radii": radii}

    if args.render_depth:
        return_pkg["depth"] = rendered_depth

    if args.render_opacity:
        return_pkg["opacity"] = rendered_alphas  # [1, H, W]

    if args.render_sky:
        # can be implemented by sky box / sky HDRI / sky MLP
        sky_bg = pc.get_sky_bg(viewpoint_camera)
        return_pkg["sky_bg"] = sky_bg # expect [3, H, W]

        # blend sky with rendered image
        if args.blend_sky:
            assert args.render_opacity
            rendered_image = rendered_image + (1 - rendered_alphas) * sky_bg

    return_pkg['render'] = rendered_image

    return return_pkg