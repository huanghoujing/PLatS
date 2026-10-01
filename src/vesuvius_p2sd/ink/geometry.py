"""Surface sampling in explicit native XYZ and array ZYX coordinates.

Reference sampling follows the pixel-centre convention documented by the pinned
villa tifxyz_label_transfer/core.py. Local refinement inherits its reference UV
chart; it is not an independent global unwrapping algorithm.
"""
from pathlib import Path
import json

import nibabel as nib
import numpy as np
from scipy.ndimage import map_coordinates, gaussian_filter
import tifffile


def sample_surface(mesh_dir, canvas_shape, origin_yx, shape_yx):
    fields=np.stack([tifffile.imread(Path(mesh_dir)/f"{a}.tif") for a in "xyz"],axis=-1)
    valid=(fields >= 0).all(-1) & np.isfinite(fields).all(-1)
    sy,sx=np.array(fields.shape[:2])/canvas_shape
    y,x=np.meshgrid((np.arange(shape_yx[0])+origin_yx[0]+.5)*sy,
                    (np.arange(shape_yx[1])+origin_yx[1]+.5)*sx,indexing="ij")

    def sample(y,x):
        if (y.min()<0 or x.min()<0 or y.max()>fields.shape[0]-1 or x.max()>fields.shape[1]-1):
            raise ValueError("Surface sample is outside the stored mesh")
        values=np.stack([map_coordinates(fields[...,i],[y,x],order=1,mode="nearest") for i in range(3)],-1)
        coverage=map_coordinates(valid.astype(float),[y,x],order=1,mode="nearest")>.99999
        return values,coverage

    xyz,good=sample(y,x)
    px,gpx=sample(y,x+1);mx,gmx=sample(y,x-1)
    py,gpy=sample(y+1,x);my,gmy=sample(y-1,x)
    normal=np.cross(px-mx,py-my)
    norm=np.linalg.norm(normal,axis=-1)
    good &= gpx & gmx & gpy & gmy & (norm>1e-6)
    normal/=np.maximum(norm[...,None],1e-6)
    return xyz,normal,good


def sample_volume(volume, xyz, origin_zyx):
    coords=xyz[...,::-1]-np.array(origin_zyx)
    good=((coords>=0)&(coords<=np.array(volume.shape)-1)).all(-1)
    values=map_coordinates(volume.astype(np.float32,copy=False),np.moveaxis(coords,-1,0),
                           order=1,mode="constant",cval=0)
    return values,good


def render(volume, xyz, normals, origin_zyx, offsets):
    stack=[];valid=np.ones(xyz.shape[:2],bool)
    for offset in offsets:
        values,good=sample_volume(volume,xyz+normals*offset,origin_zyx)
        stack.append(values);valid &= good
    return np.stack(stack),valid


def refine_surface(prob, xyz, normals, origin_zyx, radius=12, sigma=1.5):
    offsets=np.arange(-radius,radius+.25,.5)
    rays,inside=render(prob,xyz,normals,origin_zyx,offsets)
    peak=rays.argmax(axis=0);peak_offset=offsets[peak]
    # Centroid of the local probability peak, not of neighboring sheets.
    band=np.abs(offsets[:,None,None]-peak_offset)<=2
    weight=rays*band
    displacement=(weight*offsets[:,None,None]).sum(0)/np.maximum(weight.sum(0),1e-8)
    support=inside & (rays.max(0)>=.5) & (np.abs(peak_offset)<radius-1)
    smooth_weight=gaussian_filter(support.astype(float),sigma)
    smooth=gaussian_filter(displacement*support,sigma)/np.maximum(smooth_weight,1e-8)
    result=xyz+normals*smooth[...,None]
    dy,dx=np.gradient(result,axis=(0,1))
    new_normal=np.cross(dx,dy)
    norm=np.linalg.norm(new_normal,axis=-1)
    new_normal/=np.maximum(norm[...,None],1e-8)
    new_normal*=np.where((new_normal*normals).sum(-1)<0,-1,1)[...,None]
    support &= norm>1e-8
    return result,new_normal,support,smooth


def native_nifti(path, array, origin_zyx, spacing_um):
    affine=np.diag([spacing_um/1000]*3+[1.])
    affine[:3,3]=np.array(origin_zyx)[::-1]*spacing_um/1000
    nii=nib.Nifti1Image(np.ascontiguousarray(array.transpose(2,1,0)),affine)
    nii.header.set_xyzt_units("mm");nii.set_sform(affine,code=2);nii.set_qform(affine,code=0)
    nib.save(nii,path)


def geometry_stats(xyz, valid):
    dy,dx=np.gradient(xyz,axis=(0,1))
    a=(dx*dx).sum(-1);b=(dx*dy).sum(-1);c=(dy*dy).sum(-1)
    delta=np.sqrt(np.maximum((a-c)**2+4*b*b,0))
    large=np.sqrt(np.maximum((a+c+delta)/2,0));small=np.sqrt(np.maximum((a+c-delta)/2,0))
    if not valid.any():
        return {"valid_fraction":0.}
    return {"valid_fraction":float(valid.mean()),"area_voxels2":float(np.linalg.norm(np.cross(dx,dy),axis=-1)[valid].sum()),
            "stretch_min_p05":float(np.quantile(small[valid],.05)),
            "stretch_max_p95":float(np.quantile(large[valid],.95)),
            "anisotropy_p95":float(np.quantile((large/np.maximum(small,1e-8))[valid],.95))}
