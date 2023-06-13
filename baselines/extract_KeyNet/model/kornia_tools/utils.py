# TODO: update to official Kornia functions after being able to use latest version
# Copied from oficial website: https://kornia.readthedocs.io/en/latest/_modules/kornia/feature/laf.html
import kornia
import torch
import torch.nn.functional as F
from kornia.filters import filter2D
from kornia.feature.laf import normalize_laf, denormalize_laf, scale_laf, raise_error_if_laf_is_not_valid, \
    get_laf_scale, generate_patch_grid_from_normalized_LAF


def laf_from_center_scale_ori(xy: torch.Tensor, scale: torch.Tensor, ori: torch.Tensor) -> torch.Tensor:
    """Returns orientation of the LAFs, in radians. Useful to create kornia LAFs from OpenCV keypoints

    Args:
        xy: (torch.Tensor): tensor [BxNx2].
        scale: (torch.Tensor): tensor [BxNx1x1].
        ori: (torch.Tensor): tensor [BxNx1].

    Returns:
        torch.Tensor: tensor  BxNx2x3 .
    """
    names = ['xy', 'scale', 'ori']
    for var_name, var, req_shape in zip(names,
                                        [xy, scale, ori],
                                        [("B", "N", 2), ("B", "N", 1, 1), ("B", "N", 1)]):
        if not isinstance(var, torch.Tensor):
            raise TypeError("{} type is not a torch.Tensor. Got {}"
                            .format(var_name, type(var)))
        if len(var.shape) != len(req_shape):  # type: ignore  # because it does not like len(tensor.shape)
            raise TypeError(
                "{} shape should be must be [{}]. "
                "Got {}".format(var_name, str(req_shape), var.size()))
        for i, dim in enumerate(req_shape):  # type: ignore # because it wants typing for dim
            if dim is not int:
                continue
            if var.size(i) != dim:
                raise TypeError(
                    "{} shape should be must be [{}]. "
                    "Got {}".format(var_name, str(req_shape), var.size()))
    unscaled_laf: torch.Tensor = torch.cat([kornia.geometry.angle_to_rotation_matrix(ori.squeeze(-1)),
                                            xy.unsqueeze(-1)], dim=-1)
    laf: torch.Tensor = scale_laf(unscaled_laf, scale)
    return laf


def extract_patches_from_pyramid(img: torch.Tensor,
                                 laf: torch.Tensor,
                                 PS: int = 32,
                                 normalize_lafs_before_extraction: bool = True) -> torch.Tensor:
    """Extract patches defined by LAFs from image tensor.
    Patches are extracted from appropriate pyramid level

    Args:
        laf: (torch.Tensor).
        images: (torch.Tensor) images, LAFs are detected in
        PS: (int) patch size, default = 32
        normalize_lafs_before_extraction (bool):  if True, lafs are normalized to image size, default = True

    Returns:
        patches: (torch.Tensor)  :math:`(B, N, CH, PS,PS)`
    """
    raise_error_if_laf_is_not_valid(laf)
    if normalize_lafs_before_extraction:
        nlaf: torch.Tensor = normalize_laf(laf, img)
    else:
        nlaf = laf
    B, N, _, _ = laf.size()
    num, ch, h, w = img.size()
    scale = 2.0 * get_laf_scale(denormalize_laf(nlaf, img)) / float(PS)
    half: float = 0.5
    pyr_idx = (scale.log2() + half).relu().long()
    cur_img = img
    cur_pyr_level = 0
    out = torch.zeros(B, N, ch, PS, PS).to(nlaf.dtype).to(nlaf.device)
    while min(cur_img.size(2), cur_img.size(3)) >= PS:
        num, ch, h, w = cur_img.size()
        # for loop temporarily, to be refactored
        for i in range(B):
            scale_mask = (pyr_idx[i] == cur_pyr_level).squeeze()
            if (scale_mask.float().sum()) == 0:
                continue
            scale_mask = (scale_mask > 0).view(-1)
            grid = generate_patch_grid_from_normalized_LAF(
                cur_img[i:i + 1],
                nlaf[i:i + 1, scale_mask, :, :],
                PS)
            patches = F.grid_sample(cur_img[i:i + 1].expand(grid.size(0), ch, h, w), grid,  # type: ignore
                                    # padding_mode="border", align_corners=False)
                                    padding_mode="border")
            out[i].masked_scatter_(scale_mask.view(-1, 1, 1, 1), patches)
        cur_img = kornia.geometry.transform.pyrdown(cur_img)
        cur_pyr_level += 1
    return out


# Utility from Kornia: https://kornia.readthedocs.io/en/latest/_modules/kornia/geometry/transform/pyramid.html
def _get_pyramid_gaussian_kernel() -> torch.Tensor:
    """Utility function that return a pre-computed gaussian kernel."""
    return torch.tensor([[
        [1., 4., 6., 4., 1.],
        [4., 16., 24., 16., 4.],
        [6., 24., 36., 24., 6.],
        [4., 16., 24., 16., 4.],
        [1., 4., 6., 4., 1.]
    ]]) / 256.


def custom_pyrdown(input: torch.Tensor, factor: float = 2., border_type: str = 'reflect', align_corners: bool = False) -> torch.Tensor:
    r"""Blurs a tensor and downsamples it.

    Args:
        input (tensor): the tensor to be downsampled.
        border_type (str): the padding mode to be applied before convolving.
          The expected modes are: ``'constant'``, ``'reflect'``,
          ``'replicate'`` or ``'circular'``. Default: ``'reflect'``.
        align_corners(bool): interpolation flag. Default: False. See
        https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.interpolate for detail.

    Return:
        torch.Tensor: the downsampled tensor.

    Examples:
        >>> input = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4)
        >>> pyrdown(input, align_corners=True)
        tensor([[[[ 3.7500,  5.2500],
                  [ 9.7500, 11.2500]]]])
    """
    if not len(input.shape) == 4:
        raise ValueError(f"Invalid input shape, we expect BxCxHxW. Got: {input.shape}")
    kernel: torch.Tensor = _get_pyramid_gaussian_kernel()
    b, c, height, width = input.shape
    # blur image
    x_blur: torch.Tensor = filter2D(input, kernel, border_type)

    # downsample.
    out: torch.Tensor = F.interpolate(x_blur, size=(int(height // factor), int(width // factor)), mode='bilinear',
                                      align_corners=align_corners)
    return out
