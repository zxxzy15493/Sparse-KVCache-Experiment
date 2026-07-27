import argparse
import os
import time


def set_single_thread_env():
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples", type=int, default=65536)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--clusters", type=int, default=64)
    parser.add_argument("--max-iter", type=int, default=12)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--seed", type=int, default=4321)
    parser.add_argument("--cpu", type=int, default=None)
    parser.add_argument("--no-sklearnex", action="store_true")
    args = parser.parse_args()

    set_single_thread_env()
    if args.cpu is not None:
        os.sched_setaffinity(0, [args.cpu])

    import numpy as np

    if not args.no_sklearnex:
        try:
            import sklearnex

            sklearnex.patch_sklearn(verbose=False)
            print("sklearnex=patched", flush=True)
        except Exception as exc:
            print(f"sklearnex=unavailable error={exc}", flush=True)
    else:
        print("sklearnex=disabled", flush=True)

    from sklearn.cluster import KMeans

    print(
        "config "
        f"pid={os.getpid()} affinity={sorted(os.sched_getaffinity(0))} "
        f"OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')} "
        f"MKL_NUM_THREADS={os.environ.get('MKL_NUM_THREADS')} "
        f"OPENBLAS_NUM_THREADS={os.environ.get('OPENBLAS_NUM_THREADS')} "
        f"n_samples={args.n_samples} dim={args.dim} clusters={args.clusters} "
        f"max_iter={args.max_iter} repeat={args.repeat}",
        flush=True,
    )

    rng = np.random.default_rng(args.seed)
    xb = rng.standard_normal((args.n_samples, args.dim), dtype=np.float32)
    init_idx = rng.choice(np.arange(args.n_samples), size=args.clusters, replace=False)
    init = xb[init_idx].copy()

    for i in range(args.repeat):
        km = KMeans(
            n_clusters=args.clusters,
            n_init=1,
            init=init,
            tol=0.0001,
            verbose=False,
            max_iter=args.max_iter,
            random_state=0,
            algorithm="lloyd",
        )
        begin = time.perf_counter()
        result = km.fit(xb)
        elapsed = time.perf_counter() - begin
        print(
            f"run={i + 1} fit_s={elapsed:.6f} "
            f"n_iter={getattr(result, 'n_iter_', None)} "
            f"inertia={getattr(result, 'inertia_', None)}",
            flush=True,
        )


if __name__ == "__main__":
    main()
