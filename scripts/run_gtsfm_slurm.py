#!/usr/bin/env python
"""
Standalone script to run GTSFM on SLURM clusters using dask-jobqueue.

Example usage:
    python scripts/run_gtsfm_slurm.py \\
        --account myproject \\
        --queue gpu \\
        --num_jobs 4 \\
        --cores 8 \\
        --processes 4 \\
        --memory 32GB \\
        --walltime 02:00:00 \\
        -- --loader colmap --dataset_dir /path/to/data --config_name sift_front_end.yaml

    # With GPU support:
    python scripts/run_gtsfm_slurm.py \\
        --account myproject \\
        --queue gpu \\
        --gpu \\
        --gpus_per_job 2 \\
        --num_jobs 4 \\
        -- --loader colmap --dataset_dir /path/to/data --config_name sift_front_end.yaml

    # With InfiniBand and local storage:
    python scripts/run_gtsfm_slurm.py \\
        --account myproject \\
        --interface ib0 \\
        --local_directory '$TMPDIR' \\
        --num_jobs 8 \\
        -- --loader colmap --dataset_dir /path/to/data --config_name sift_front_end.yaml
"""

import argparse
import os
import sys
import time

import gtsfm.utils.logger as logger_utils

# Add gtsfm to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logger_utils.get_logger()


def create_slurm_cluster(args):
    """Create and configure SLURM cluster using dask-jobqueue.

    Args:
        args: Parsed command-line arguments

    Returns:
        SLURMCluster instance
    """
    try:
        from dask_jobqueue import SLURMCluster
    except ImportError:
        raise ImportError(
            "dask-jobqueue is required for SLURM clusters.\n"
            "Install with: conda install dask-jobqueue"
        )

    # Build cluster configuration
    cluster_kwargs = {
        "cores": args.cores,
        "processes": args.processes,
        "memory": args.memory,
        "walltime": args.walltime,
        "queue": args.queue,
        "account": args.account,
    }

    # Optional parameters
    if args.interface:
        cluster_kwargs["interface"] = args.interface
        logger.info(f"🌐 Network interface: {args.interface}")

    if args.local_directory:
        cluster_kwargs["local_directory"] = args.local_directory
        logger.info(f"💾 Local directory: {args.local_directory}")

    if args.dashboard_port:
        cluster_kwargs["dashboard_address"] = args.dashboard_port

    # Handle GPU configuration
    if args.gpu:
        job_extra = []
        if args.job_extra:
            job_extra.extend(args.job_extra)
        job_extra.append(f"--gres=gpu:{args.gpus_per_job}")
        cluster_kwargs["job_extra_directives"] = job_extra
        cluster_kwargs["worker_extra_args"] = [f"--resources GPU={args.gpus_per_job}"]
        logger.info(f"🎮 GPU mode: {args.gpus_per_job} GPU(s) per job")
    elif args.job_extra:
        cluster_kwargs["job_extra_directives"] = args.job_extra

    logger.info("🚀 Creating SLURM cluster:")
    logger.info(f"   Account: {args.account}")
    logger.info(f"   Queue: {args.queue}")
    logger.info(f"   Cores: {args.cores}, Processes: {args.processes}, Memory: {args.memory}")
    logger.info(f"   Walltime: {args.walltime}")

    cluster = SLURMCluster(**cluster_kwargs)
    cluster.scale(jobs=args.num_jobs)
    logger.info(f"📈 Scaling cluster to {args.num_jobs} SLURM jobs")

    return cluster


def wait_for_workers(client, min_workers=1, timeout=300):
    """Wait for at least min_workers to connect.

    Args:
        client: Dask client
        min_workers: Minimum number of workers to wait for
        timeout: Maximum time to wait in seconds

    Returns:
        Number of workers connected
    """
    logger.info(f"⏳ Waiting for at least {min_workers} worker(s) to start (timeout: {timeout}s)...")
    start_time = time.time()

    while True:
        n_workers = len(client.scheduler_info()["workers"])
        if n_workers >= min_workers:
            logger.info(f"✅ {n_workers} worker(s) connected and ready")
            return n_workers

        elapsed = time.time() - start_time
        if elapsed > timeout:
            logger.warning(
                f"⚠️ Timeout after {timeout}s. Only {n_workers} worker(s) connected. "
                "Proceeding anyway - more workers may join later."
            )
            logger.info("💡 Check SLURM queue status with: squeue -u $USER")
            return n_workers

        if n_workers > 0:
            logger.info(f"   {n_workers} worker(s) connected so far, waiting for more...")
        time.sleep(5)


def main():
    parser = argparse.ArgumentParser(
        description="Run GTSFM on SLURM cluster",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # SLURM-specific arguments
    parser.add_argument("--account", required=True, help="SLURM account/project name")
    parser.add_argument("--queue", default="normal", help="SLURM partition/queue (default: normal)")
    parser.add_argument("--walltime", default="01:00:00", help="Job walltime HH:MM:SS (default: 01:00:00)")
    parser.add_argument("--num_jobs", type=int, default=4, help="Number of SLURM jobs to submit (default: 4)")
    parser.add_argument("--cores", type=int, default=8, help="Cores per job (default: 8)")
    parser.add_argument("--processes", type=int, default=4, help="Dask processes per job (default: 4)")
    parser.add_argument("--memory", default="32GB", help="Memory per job (default: 32GB)")

    # Optional SLURM arguments
    parser.add_argument("--interface", help="Network interface (e.g., ib0 for InfiniBand)")
    parser.add_argument("--local_directory", help="Local temp directory (e.g., $TMPDIR or /scratch/$USER)")
    parser.add_argument("--job_extra", action="append", help="Extra SLURM directives (repeatable)")

    # GPU arguments
    parser.add_argument("--gpu", action="store_true", help="Request GPU resources")
    parser.add_argument("--gpus_per_job", type=int, default=1, help="Number of GPUs per job (default: 1)")

    # Dask dashboard
    parser.add_argument("--dashboard_port", default=":8787", help="Dask dashboard port (default: :8787)")

    # Timeout for waiting for workers
    parser.add_argument(
        "--worker_timeout", type=int, default=300, help="Timeout for waiting for workers in seconds (default: 300)"
    )
    parser.add_argument(
        "--min_workers", type=int, default=1, help="Minimum workers to wait for before starting (default: 1)"
    )

    args, gtsfm_args = parser.parse_known_args()

    # Validate arguments
    if not gtsfm_args:
        parser.error("No GTSFM arguments provided. Use -- to separate SLURM args from GTSFM args.")

    logger.info("=" * 80)
    logger.info("GTSFM SLURM Launcher")
    logger.info("=" * 80)

    # Create SLURM cluster
    try:
        cluster = create_slurm_cluster(args)
    except Exception as e:
        logger.error(f"❌ Failed to create SLURM cluster: {e}")
        return 1

    # Create Dask client
    try:
        from dask.distributed import Client

        logger.info("🔗 Connecting to cluster...")
        client = Client(cluster)
        client.forward_logging()
        logger.info(f"🚀 Dask Dashboard: {client.dashboard_link}")
        logger.info("💡 Monitor jobs with: squeue -u $USER")

        # Wait for workers
        n_workers = wait_for_workers(client, min_workers=args.min_workers, timeout=args.worker_timeout)

        if n_workers == 0:
            logger.error("❌ No workers connected. Check SLURM logs and queue status.")
            return 1

    except Exception as e:
        logger.error(f"❌ Failed to connect to cluster: {e}")
        cluster.close()
        return 1

    # Create and run GTSFM
    try:
        from gtsfm.runner import GtsfmRunner

        logger.info("=" * 80)
        logger.info("Starting GTSFM")
        logger.info("=" * 80)

        # Create GTSFM runner with remaining args
        runner = GtsfmRunner(override_args=gtsfm_args)

        # Run GTSFM with the SLURM client
        logger.info("🌟 GTSFM: Starting SceneOptimizer...")
        runner.scene_optimizer.run(client)

        logger.info("✅ GTSFM completed successfully")

    except KeyboardInterrupt:
        logger.info("⚠️ Interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"❌ GTSFM execution failed: {e}")
        import traceback

        traceback.print_exc()
        return 1
    finally:
        # Cleanup
        logger.info("🧹 Cleaning up...")
        try:
            client.close()
            cluster.close()
            logger.info("✅ Cleanup complete")
        except Exception as e:
            logger.warning(f"⚠️ Cleanup warning: {e}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
