"""reg_common.py -- Registration commons

Field conventions (H, W, 2): [..., 0] = dx, [..., 1] = dy; sampling
convention output[x, y] = source[x - dy, y - dx].
Provides: DIS dense optical flow, guarded ECC translation, remap-based
warp for images/masks, field composition, affine-to-dense conversion,
NCC and field statistics.
"""
import cv2
import numpy as np
import scipy.ndimage as ndimage

H, W = 647, 549
_YY, _XX = np.mgrid[0:H, 0:W].astype(np.float32)


# ============================================================
def warp_image(image, disp, order=1):
    """output[x,y] = source[x-dy, y-dx]"""
    return ndimage.map_coordinates(
        image.astype(np.float32),
        [_YY - disp[..., 1], _XX - disp[..., 0]],
        order=order, mode="constant", cval=0.0,
    ).astype(np.float32)


def warp_mask(mask_bool, disp):
    """binary maskrandom displacement field (order=0), return bool"""
    w = warp_image(mask_bool.astype(np.float32), disp, order=0)
    return w > 0.5


def compose(A, B):
    """A first apply for source , B for in between results (A: frame →anchor , B: anchor → )
     into displacement : d(x) = B(x) + A(x − B(x))"""
    ax = warp_image(A[..., 0], B)
    ay = warp_image(A[..., 1], B)
    out = np.empty_like(A)
    out[..., 0] = B[..., 0] + ax
    out[..., 1] = B[..., 1] + ay
    return out


def compose_affine_dense(M, F):
    """affine ( , cv2.warpAffine use ) + (after ) displacement field
    raw x: source sampling position = M·(x − F(x)) ⇒ d(x) = x − M·(x − F(x))
    """
    x = _XX - F[..., 0]
    y = _YY - F[..., 1]
    mx = M[0, 0] * x + M[0, 1] * y + M[0, 2]
    my = M[1, 0] * x + M[1, 1] * y + M[1, 2]
    out = np.empty_like(F)
    out[..., 0] = _XX - mx
    out[..., 1] = _YY - my
    return out


def affine_to_field(M, shape=(H, W)):
    """2x3 affine matrix → displacement field A(x) = M·[x,y,1] − [x,y]"""
    grid = np.stack([_XX, _YY, np.ones_like(_XX)], -1)  # (H,W,3)
    warped = np.einsum("ij,hwj->hwi", M, grid)
    field = np.empty(shape + (2,), dtype=np.float32)
    field[..., 0] = warped[..., 0] - _XX
    field[..., 1] = warped[..., 1] - _YY
    return field


# ============================================================
def to_u8(img):
    if img.dtype == np.uint8:
        return img
    if img.dtype == np.float32 or img.dtype == np.float64:
        if float(np.nanmax(img)) > 1.5:  # 0-255 point
            img = img / 255.0
        return (np.clip(img, 0, 1) * 255).astype(np.uint8)
    return (np.clip(img.astype(np.float32), 0, 1) * 255).astype(np.uint8)


def dis_flow(moving, fixed, prev=None, preset="MEDIUM"):
    """DIS moving→fixed, return displacement field (H,W,2)
    displacement with : as make moving for fixed, pixel (x,y) take self moving (x-dx, y-dy)
    """
    dis = cv2.DISOpticalFlow_create(getattr(cv2, f"DISOPTICAL_FLOW_PRESET_{preset}"))
    if prev is not None:
        flow = dis.calc(to_u8(moving), to_u8(fixed), prev.copy())
    else:
        flow = dis.calc(to_u8(moving), to_u8(fixed), None)
    return flow.astype(np.float32)


def affine_ecc(moving, fixed, iterations=50, eps=1e-6):
    """ECC (rotation +translation ) moving→fixed, return 2x3 matrix ; failure return None"""
    m, f = to_u8(moving), to_u8(fixed)
    warp = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, iterations, eps)
    try:
        cc, warp = cv2.findTransformECC(
            f, m, warp, cv2.MOTION_EUCLIDEAN, criteria, None, 5)
    except cv2.error:
        return None
    if cc is None or cc < 0:
        return None
    return warp


def apply_affine(img, M):
    """set affine matrix use to image (cv2.warpAffine, output and input same shape )"""
    return cv2.warpAffine(img.astype(np.float32), M, (W, H),
                          flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                          borderValue=0).astype(np.float32)


def affine_ecc_guarded(moving, fixed, mask=None, smooth_sigma=4.0, max_shift=10.0,
                       min_ncc_gain=-0.005):
    """ translation (speckle data on ECC rotation not , translation ):
    1. smoothing σ=smooth_sigma (speckle height noise down ECC )
    2. translation (MOTION_TRANSLATION)
    3. : |translation | > max_shift or correction after NCC not ( < min_ncc_gain) → etc.
    return (2x3matrix , is )
    """
    m, f = to_u8(moving), to_u8(fixed)
    m_s = cv2.GaussianBlur(m, (0, 0), smooth_sigma)
    f_s = cv2.GaussianBlur(f, (0, 0), smooth_sigma)
    warp = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1e-6)
    try:
        cc, warp = cv2.findTransformECC(f_s, m_s, warp, cv2.MOTION_TRANSLATION,
                                        criteria, None, 5)
    except cv2.error:
        return np.eye(2, 3, dtype=np.float32), False
    if cc is None or cc < 0:
        return np.eye(2, 3, dtype=np.float32), False
    tx, ty = float(warp[0, 2]), float(warp[1, 2])
    if abs(tx) > max_shift or abs(ty) > max_shift:
        return np.eye(2, 3, dtype=np.float32), False
    # NCC verify (heart region )
    before = ncc(m.astype(np.float32) / 255.0, f.astype(np.float32) / 255.0, mask)
    after = ncc(apply_affine(m.astype(np.float32) / 255.0, warp),
                f.astype(np.float32) / 255.0, mask)
    if after < before + min_ncc_gain:
        return np.eye(2, 3, dtype=np.float32), False
    return warp, True


# ============================================================
def ncc(a, b, mask=None):
    if mask is not None:
        a, b = a[mask], b[mask]
    sa, sb = a.std(), b.std()
    if sa < 1e-9 or sb < 1e-9:
        return 0.0
    return float(np.corrcoef(a.ravel(), b.ravel())[0, 1])


def field_stats(disp):
    """displacement field statistics : mean |displacement |, folding ratio (det J < 0), most large |displacement |"""
    mag = np.sqrt(disp[..., 0] ** 2 + disp[..., 1] ** 2)
    dFx_dx, dFx_dy = np.gradient(disp[..., 0])
    dFy_dx, dFy_dy = np.gradient(disp[..., 1])
    detJ = (1 + dFx_dx) * (1 + dFy_dy) - dFx_dy * dFy_dx
    return {
        "mean_mag": float(mag.mean()),
        "p99_mag": float(np.percentile(mag, 99)),
        "fold_ratio": float((detJ < 0).mean()),
    }


def forward_scatter_mask(ref_mask, disp):
    """ empty between mask → frame empty between mask ( , )
    M_t[y+dy, x+dx] = M_ref[y, x] (take most , ensure ROI pixel self mask, statistics use raw frame value )
    """
    out = np.zeros((H, W), dtype=bool)
    ys, xs = np.where(ref_mask)
    ty = np.clip(np.round(ys + disp[ys, xs, 1]).astype(np.int64), 0, H - 1)
    tx = np.clip(np.round(xs + disp[ys, xs, 0]).astype(np.int64), 0, W - 1)
    out[ty, tx] = True
    return out
