import os
import glob
import argparse

import numpy as np
import nibabel as nib

from probreg import cpd
from tqdm import tqdm


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


def match(pc_ref, pc_pred, cpd_type, num_sample, match_sigma, use_cuda):
    # num_sample required to avoid OOM
    # center both point clouds to make it easier to match
    pc_ref_mean = np.mean(pc_ref, axis=0)
    pc_pred_mean = np.mean(pc_pred, axis=0)
    pc_ref = pc_ref - pc_ref_mean
    pc_pred = pc_pred - pc_pred_mean

    # cpd registration
    # allow sampling with replacement just in case len(pc) < num_sample
    pc_ref_sample = pc_ref[np.random.choice(pc_ref.shape[0], num_sample, replace=True)]
    pc_pred_sample = pc_pred[
        np.random.choice(pc_pred.shape[0], num_sample, replace=True)
    ]

    # add noise, to prevent regularly spaced points
    pc_ref_sample += np.random.normal(0, match_sigma, pc_ref_sample.shape)
    pc_pred_sample += np.random.normal(0, match_sigma, pc_pred_sample.shape)

    tf_param, _, _ = cpd.registration_cpd(
        pc_ref_sample, pc_pred_sample, tf_type_name=cpd_type, use_cuda=use_cuda
    )
    transformed = tf_param.transform(pc_ref) + pc_pred_mean

    return transformed


def densify(pc, num_duplicate, densify_sigma):
    points = [pc]
    for _ in range(num_duplicate):
        points.append(pc + np.random.normal(0, densify_sigma, pc.shape))
    return np.concatenate(points, axis=0)


def voxelize(shape, pc):
    result = np.zeros(shape, dtype=int)
    pc = np.round(pc).astype(int)
    pc = np.clip(pc, 0, np.array(shape) - 1)
    result[pc[:, 0], pc[:, 1], pc[:, 2]] = 1

    return result


def post_process(
    reference,
    prediction,
    cpd_type,
    num_sample,
    match_sigma,
    use_cuda,
    num_duplicate,
    densify_sigma,
):
    result = np.zeros(prediction.shape, dtype=int)
    for i in tqdm(range(1, len(ORDERING) + 1), leave=False):
        ref_pc = np.array(np.where(reference == i)).T.astype(float)
        pred_pc = np.array(np.where(prediction == i)).T.astype(float)

        if len(pred_pc) == 0:
            continue

        # deform reference to prediction
        transformed_pc = match(
            ref_pc, pred_pc, cpd_type, num_sample, match_sigma, use_cuda
        )
        densified_pc = densify(transformed_pc, num_duplicate, densify_sigma)
        segmentation = voxelize(prediction.shape, densified_pc)

        result = result * (segmentation == 0) + segmentation * i
    return result


def main(
    reference_path,
    predict_path,
    output_path,
    cpd_type,
    num_sample,
    match_sigma,
    use_cuda,
    num_duplicate,
    densify_sigma,
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
            num_sample,
            match_sigma,
            use_cuda,
            num_duplicate,
            densify_sigma,
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
    parser.add_argument("--num_sample", type=int, default=2048)
    parser.add_argument("--match_sigma", type=int, default=1.0)
    parser.add_argument("--cpd_type", type=str, default="nonrigid")
    parser.add_argument("--use_cuda", action="store_true")  # requires cupy
    parser.add_argument("--num_duplicate", type=int, default=4)
    parser.add_argument("--densify_sigma", type=int, default=1)

    args = parser.parse_args()

    assert args.cpd_type in ["rigid", "affine", "nonrigid", "nonrigid_constrained"]

    main(
        args.reference_path,
        args.predict_path,
        args.output_path,
        args.cpd_type,
        args.num_sample,
        args.match_sigma,
        args.use_cuda,
        args.num_duplicate,
        args.densify_sigma,
    )
