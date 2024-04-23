import os
import glob
import argparse

import numpy as np
import nibabel as nib

from probreg import cpd, bcpd
from tqdm import tqdm

import pyvista as pv
import pymeshfix as mf

"""
reference was obtained from Totalsegmentator_dataset/s0224, a case which had complete vertebrae annotations
"""

ORDERING = [
    "L5",
    "L4",
    "L3",
    "L2",
    "L1",
    "T12",
    "T11",
    "T10",
    "T9",
    "T8",
    "T7",
    "T6",
    "T5",
    "T4",
    "T3",
    "T2",
    "T1",
    "C7",
    "C6",
    "C5",
    "C4",
    "C3",
    "C2",
    "C1",
]


def read_reference(reference_path):
    reference = {}
    for file in sorted(glob.glob(os.path.join(reference_path, "*"))):
        # from path .../vertebrae_L1.nii.gz to L1
        segment = os.path.basename(file).split(".")[0].split("_")[1]
        reference[segment] = nib.load(file).get_fdata() > 0
    assert len(set([reference[x].shape for x in ORDERING])) == 1
    label = np.zeros(reference[ORDERING[0]].shape, dtype=int)
    for i, segment in enumerate(ORDERING):
        label[reference[segment]] = i + 1
    return label


def align_mesh(ref_mesh, pred_mesh, cpd_type, target_reduction, cpd_kwargs):
    # takes in ref_mesh and pred_mesh, returns aligned ref_mesh
    # NOTE: modifies ref_mesh and pred_mesh in place

    ref_mean = ref_mesh.points.mean(axis=0)
    pred_mean = pred_mesh.points.mean(axis=0)
    ref_std = ref_mesh.points.std(axis=0)
    pred_std = pred_mesh.points.std(axis=0)

    # normalize to unit sphere
    ref_mesh.points = (ref_mesh.points - ref_mean) / ref_std
    pred_mesh.points = (pred_mesh.points - pred_mean) / pred_std

    # decimate to reduce number of triangle vertices
    ref_mesh = ref_mesh.decimate(target_reduction)
    pred_mesh = pred_mesh.decimate(target_reduction)

    # ensure watertightness; applying the transformation might cause overlapping triangles, and then mesh.repair() will delete all the faces, nuking the mesh
    ref_mesh = fix_mesh(ref_mesh)
    pred_mesh = fix_mesh(pred_mesh)

    tf_param, _, _ = cpd.registration_cpd(
        ref_mesh.points,
        pred_mesh.points,
        tf_type_name=cpd_type,
        **cpd_kwargs,
    )

    aligned_ref_pc = tf_param.transform(ref_mesh.points) * pred_std + pred_mean
    # tf_param = bcpd.registration_bcpd(ref_mesh.points, pred_mesh.points)
    # aligned_ref_pc = tf_param.transform(ref_mesh.points) * pred_std + pred_mean

    ref_mesh.points = aligned_ref_pc

    return ref_mesh


def fix_mesh(mesh, joincomp=False, remove_smallest_components=True):
    mesh_fix = mf.MeshFix(mesh)
    mesh_fix.repair(
        joincomp=joincomp,
        remove_smallest_components=remove_smallest_components,
        verbose=True,
    )
    if mesh_fix.mesh.points.shape[0] < 200:
        raise ValueError("MeshFix nuked mesh")
    return mesh_fix.mesh


def voxel_to_mesh(vol):
    # https://docs.pyvista.org/version/stable/examples/00-load/create-uniform-grid.html
    grid = pv.ImageData()
    grid.dimensions = vol.shape
    grid.point_data["values"] = vol.flatten(order="F")

    # create mesh by running flying edges with isocontour value of 0.5
    mesh = grid.contour([0.5])
    return mesh


def mesh_to_voxel(mesh, shape):
    # crashes if check_surface is True, very weird
    grid = pv.voxelize(mesh, density=1, check_surface=False)
    # flooring should be fine, see https://github.com/pyvista/pyvista/blob/392d29ea4398c7292275300b44b75bcdbbed1c2e/pyvista/core/utilities/features.py#L78-L82
    points = np.floor(grid.points).astype(int)
    points = np.clip(points, 0, np.array(shape) - 1)

    result = np.zeros(shape, dtype=int)
    result[points[:, 0], points[:, 1], points[:, 2]] = 1

    return result


def post_process(reference, prediction, cpd_type, target_reduction, cpd_kwargs):
    result = np.zeros(prediction.shape, dtype=int)
    for i in tqdm(range(1, len(ORDERING) + 1), leave=False):
        # if component missing, just skip it
        if np.sum(prediction == i) == 0:
            continue

        ref_mesh = voxel_to_mesh(reference == i)
        pred_mesh = voxel_to_mesh(prediction == i)

        # deform reference to prediction
        transformed_mesh = align_mesh(
            ref_mesh, pred_mesh, cpd_type, target_reduction, cpd_kwargs
        )
        # transformed_mesh = fix_mesh(transformed_mesh)

        segmentation = mesh_to_voxel(transformed_mesh, prediction.shape)

        result = result * (segmentation == 0) + segmentation * i
    return result


def main(
    reference_path, predict_path, output_path, cpd_type, target_reduction, cpd_kwargs
):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    reference = read_reference(reference_path)
    for folder in tqdm(sorted(glob.glob(os.path.join(predict_path, "*")))):
        nib_file = nib.load(os.path.join(folder, "combined_labels.nii.gz"))
        prediction = nib_file.get_fdata().astype(int)
        result = post_process(
            reference,
            prediction,
            cpd_type,
            target_reduction,
            cpd_kwargs,
        )

        name = os.path.basename(folder)
        if not os.path.exists(os.path.join(output_path, name)):
            os.makedirs(os.path.join(output_path, name))
        nib.save(
            nib.Nifti1Image(result.astype(np.uint8), affine=nib_file.affine),
            os.path.join(output_path, os.path.basename(name), "combined_labels.nii.gz"),
        )


if __name__ == "__main__":
    # parse arguments for paths
    parser = argparse.ArgumentParser()
    # set default
    parser.add_argument(
        "--reference_path", type=str, default="/data/adhinart/zongwei/sample_vertebrae"
    )
    parser.add_argument(
        "--predict_path",
        type=str,
        default="/data/adhinart/zongwei/AbdomenAtlasDemoPredict",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="/data/adhinart/zongwei/AbdomenAtlasDemoPredictPostProcessed",
    )
    parser.add_argument("--cpd_type", type=str, default="nonrigid")
    parser.add_argument("--use_cuda", action="store_true")  # requires cupy
    # with 0.9, ~1000 points are used for matching
    parser.add_argument(
        "--target_reduction", type=int, default=0.9
    )  # simplifying number of vertices in mesh

    cpd_kwargs = {"lmd": 1e-3, "beta": 1e-3}

    args = parser.parse_args()

    assert args.cpd_type in ["rigid", "affine", "nonrigid", "nonrigid_constrained"]

    main(
        args.reference_path,
        args.predict_path,
        args.output_path,
        args.cpd_type,
        args.target_reduction,
        cpd_kwargs,
    )
