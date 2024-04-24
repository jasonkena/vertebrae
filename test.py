import numpy as np

from probreg import cpd, bcpd, filterreg

import pyvista as pv
import pymeshfix as mf

import math
from post_processing import fix_mesh, normalize_mesh


def register_cpd(ref_mesh, pred_mesh, cpd_type, **kwargs):
    ref_mesh = ref_mesh.copy()
    assert cpd_type in ["rigid", "affine", "nonrigid", "nonrigid_constrained"]

    tf_param, _, _ = cpd.registration_cpd(
        ref_mesh.points, pred_mesh.points, tf_type_name=cpd_type, **kwargs
    )

    aligned_ref_pc = tf_param.transform(ref_mesh.points)
    ref_mesh.points = aligned_ref_pc

    return ref_mesh


def register_bcpd(ref_mesh, pred_mesh, **kwargs):
    ref_mesh = ref_mesh.copy()
    tf_param = bcpd.registration_bcpd(ref_mesh.points, pred_mesh.points)
    aligned_ref_pc = tf_param.transform(ref_mesh.points)
    ref_mesh.points = aligned_ref_pc

    return ref_mesh


def register_filterreg(ref_mesh, pred_mesh, **kwargs):
    source = np.asarray(pred_mesh.points)
    target = np.asarray(ref_mesh.points)
    reg = filterreg.DeformableKinematicFilterReg(source, ws, 0.01)


def plot_results(pred_mesh, results):
    # ceil sqrt
    dim = math.ceil(math.sqrt(len(results)))
    plotter = pv.Plotter(shape=(dim, dim))
    for i, (name, mesh) in enumerate(results.items()):
        plotter.subplot(i % dim, i // dim)
        plotter.add_text(name, font_size=10)
        plotter.add_mesh(mesh, color="red", opacity=0.5)
        plotter.add_mesh(pred_mesh, color="blue", opacity=0.5)
    plotter.show()


def main():
    sample = np.load("broken_6.npz", allow_pickle=True)
    # sample = np.load("sample.npz", allow_pickle=True)
    ref_mesh = sample["ref_mesh"].item()
    pred_mesh = sample["pred_mesh"].item()

    # results = {}
    # results["original"] = ref_mesh
    # for cpd_type in ["rigid", "affine", "nonrigid", "nonrigid_constrained"]:
    #     results[cpd_type] = register_cpd(ref_mesh, pred_mesh, cpd_type)
    # results["bcpd"] = register_bcpd(ref_mesh, pred_mesh)

    # plot_results(pred_mesh, results)

    results = {}
    cpd_type = "nonrigid"
    w = 0.1
    # cpd_type = "nonrigid_constrained"
    counter = 0

    def debug(*args, **kwargs):
        nonlocal counter
        counter += 1

    results["original"] = ref_mesh

    # max_iters = range(50, 201, 50)
    # for iters in max_iters:
    #     counter = 0
    #     results[f"{cpd_type}_{iters}"] = register_cpd(ref_mesh, pred_mesh, cpd_type, maxiter=iters, callbacks=[debug])
    #     print(f"number of iterations: {counter}")
    # plot_results(pred_mesh, results)

    # max_iter = 200
    # tol = [1e-3, 1e-4, 1e-5]
    # for t in tol:
    #     counter = 0
    #     results[f"{cpd_type}_{t}"] = register_cpd(ref_mesh, pred_mesh, cpd_type, tol=t, maxiter=max_iter, callbacks=[debug])
    #     print(f"number of iterations: {counter}")
    # plot_results(pred_mesh, results)

    # lmds = np.linspace(2, 10, 5)
    lmds = np.logspace(-4, 0, 5)
    betas = np.logspace(-4, 0, 5)
    # betas = np.linspace(2, 10, 5)
    # lmds = np.linspace(100, 1000, 5)
    # betas = np.linspace(100, 1000, 3)
    # lmds = [1e-4, 1e-3, 1e-2, 1e-1]
    # betas = [1e-4, 1e-3, 1e-2, 1e-1]
    max_iter = 200
    for l in lmds:
        for b in betas:
            counter = 0
            results[f"{cpd_type}_{l}_{b}"] = register_cpd(
                ref_mesh,
                pred_mesh,
                cpd_type,
                lmd=l,
                beta=b,
                maxiter=max_iter,
                callbacks=[debug],
                w=w,
            )
            print(f"number of iterations: {counter}")
    plot_results(pred_mesh, results)

    # lmd = 1000
    # beta = 1
    # ws = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    # max_iter = 200
    # for w in ws:
    #     counter = 0
    #     results[f"{cpd_type}_{w}"] = register_cpd(ref_mesh, pred_mesh, cpd_type, lmd=lmd, beta=beta, maxiter=max_iter, callbacks=[debug], w=w)
    #     print(f"number of iterations: {counter}")
    # plot_results(pred_mesh, results)


def fix_broken():
    sample = np.load("broken_case.npz", allow_pickle=True)
    ref_mesh = sample["ref_mesh"].item()
    pred_mesh = sample["pred_mesh"].item()

    target_reduction = 0.9
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

    tf_param, _, _ = cpd.registration_cpd(
        ref_mesh.points,
        pred_mesh.points,
        tf_type_name="nonrigid",
        lmd=0.001,
        beta=0.001,
    )
    aligned_ref_pc = tf_param.transform(ref_mesh.points)  # * pred_std + pred_mean
    ref_mesh.points = aligned_ref_pc

    # ref_mesh = fix_mesh(ref_mesh)

    # points = pv.wrap(ref_mesh.points)
    # surf = points.reconstruct_surface()
    # __import__('pdb').set_trace()
    # delaunay = ref_mesh.delaunay_2d().plot()
    mesh_fix = mf.MeshFix(ref_mesh)
    holes = mesh_fix.extract_holes()
    __import__("pdb").set_trace()
    # mesh_fix.repair(verbose=True)
    plotter = pv.Plotter()
    plotter.add_mesh(ref_mesh, color="red", opacity=0.5)
    plotter.add_mesh(holes, color="green", line_width=5)
    plotter.enable_eye_dome_lighting()
    # plotter.add_mesh(transformed_mesh, color="red", opacity=0.5)
    # plotter.add_mesh(pred_mesh, color="blue", opacity=0.5)
    plotter.show()

    # mesh_fix = mf.MeshFix(mesh)
    # tin = mf.PyTMesh()
    # # https://pymeshfix.pyvista.org/_autosummary/pymeshfix.PyTMesh.html
    # tin.load_array(mesh_fix.v, mesh_fix.f)
    # print('There are {:d} boundaries'.format(tin.boundaries()))
    # welp=tin.fill_small_boundaries(refine=False)
    # __import__('pdb').set_trace()
    # print('There are {:d} boundaries'.format(tin.boundaries()))
    # # tin.clean(max_iters=10, inner_loops=3)
    # # print('There are {:d} boundaries'.format(tin.boundaries()))
    # v, f = tin.return_arrays()
    # triangles = np.empty((f.shape[0], 4), dtype=int)
    # triangles[:, -3:] = f
    # triangles[:, 0] = 3
    # #
    # return pv.PolyData(v, triangles)


def predecimated():
    sample = np.load("predecimated.npz", allow_pickle=True)
    ref_mesh = sample["ref_mesh"].item()
    pred_mesh = sample["pred_mesh"].item()
    __import__("pdb").set_trace()
    ref_mesh = fix_mesh(ref_mesh)  # .decimate(0.9)
    pred_mesh = fix_mesh(pred_mesh)  # .decimate(0.9)
    __import__("pdb").set_trace()

    ref_mesh = normalize_mesh(ref_mesh, 3, 2000)
    pred_mesh = normalize_mesh(pred_mesh, 3, 2000)

    ref_mesh.plot()
    pred_mesh.plot()


if __name__ == "__main__":
    # main()
    # fix_broken()
    predecimated()
