#!/usr/bin/env python3
"""Utility script for setting up passwordless SSH between cluster workers.

This script automates the process of configuring passwordless SSH authentication
between all machines in a cluster by reading the cluster configuration from a YAML file,
generating SSH keys if needed, and distributing them to all workers.

Authors: Zong
"""
import subprocess
import sys
from pathlib import Path

import yaml  # type: ignore

import gtsfm.utils.logger as logger_utils

logger = logger_utils.get_logger()

# Default cluster config path relative to this script
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent  # Go up from gtsfm/utils/ to project root
DEFAULT_CLUSTER_CONFIG = PROJECT_ROOT / "gtsfm" / "configs" / "cluster.yaml"


def run_command(cmd, check=True):
    """Run a shell command and return the result.

    Args:
        cmd: Shell command string to execute.
        check: If True, log errors when command fails. Defaults to True.

    Returns:
        subprocess.CompletedProcess: Result object containing return code, stdout, and stderr.
    """
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if check and result.returncode != 0:
        logger.error(f"Command failed: {result.stderr}")
    return result


def run_interactive_command(cmd):
    """Run an interactive shell command that requires user input.

    Args:
        cmd: Shell command string to execute.

    Returns:
        int: Return code of the command.
    """
    result = subprocess.run(cmd, shell=True)
    return result.returncode


def setup_passwordless_ssh(cluster_config_path, username):
    """Setup passwordless SSH between all cluster workers.

    This function performs the following steps:
    1. Loads the cluster configuration from the specified YAML file
    2. For each machine in the cluster:
       - Checks if an SSH key exists, generates one if missing
       - Copies the SSH key to all other machines (including itself)
    3. Tests all SSH connections to verify passwordless access

    Args:
        cluster_config_path: Path to the cluster configuration YAML file containing
                           a 'workers' list with machine hostnames/IPs.
        username: Username to use for SSH connections on all machines.

    Returns:
        None. Logs progress and results, exits with code 1 on fatal errors.
    """

    # Read cluster configuration
    logger.info(f"Loading cluster configuration from: {cluster_config_path}")
    try:
        with open(cluster_config_path, "r") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        logger.error(f"Cluster config file not found: {cluster_config_path}")
        sys.exit(1)
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file: {e}")
        sys.exit(1)

    machines = config.get("workers", [])

    if not machines:
        logger.error("No workers found in cluster configuration")
        sys.exit(1)

    logger.info("=" * 50)
    logger.info("Setting up passwordless SSH for cluster")
    logger.info("=" * 50)
    logger.info(f"Machines: {machines}")
    logger.info(f"Username: {username}")
    logger.info(f"Total operations: {len(machines)} × {len(machines)} = {len(machines)**2}")
    logger.info("")
    logger.info("NOTE: You will be prompted for passwords multiple times during this process.")
    logger.info("")

    input("Press Enter to continue...")

    # Setup keys on each machine
    for source_machine in machines:
        logger.info(f"\n{'='*50}")
        logger.info(f"Setting up SSH keys on: {source_machine}")
        logger.info("=" * 50)

        # Check and generate SSH key if needed
        check_key_cmd = f"ssh {username}@{source_machine} '[ -f ~/.ssh/id_rsa ] && echo exists || echo missing'"
        result = run_command(check_key_cmd, check=False)

        if "missing" in result.stdout:
            logger.info(f"Generating SSH key on {source_machine}...")
            # Use -t for TTY allocation to allow interactive key generation
            gen_key_cmd = f"ssh -t {username}@{source_machine} 'ssh-keygen -t rsa -N \"\" -f ~/.ssh/id_rsa'"
            run_interactive_command(gen_key_cmd)
        else:
            logger.info(f"SSH key already exists on {source_machine}")

        # Copy key to all machines (interactive - will prompt for passwords)
        for target_machine in machines:
            logger.info(f"\nCopying key from {source_machine} to {target_machine}...")
            logger.info(f"  (You may be prompted for password for {username}@{target_machine})")
            # Use -t for TTY allocation to enable interactive password prompts in nested SSH
            copy_cmd = (
                f"ssh -t {username}@{source_machine} "
                f"'ssh-copy-id -o StrictHostKeyChecking=no -o ConnectTimeout=5 {username}@{target_machine}'"
            )
            returncode = run_interactive_command(copy_cmd)

            if returncode == 0:
                logger.info(f"  ✓ Successfully copied key from {source_machine} to {target_machine}")

            if returncode == 0:
                logger.info(f"  ✓ Successfully copied key from {source_machine} to {target_machine}")
            else:
                logger.warning(f"  ✗ Failed to copy key from {source_machine} to {target_machine}")

    # Test all connections
    logger.info(f"\n{'='*50}")
    logger.info("Testing passwordless SSH connections...")
    logger.info("=" * 50)

    all_success = True
    for source_machine in machines:
        for target_machine in machines:
            test_cmd = (
                f"ssh {username}@{source_machine} "
                f"'ssh -o BatchMode=yes -o ConnectTimeout=5 {username}@{target_machine} echo SUCCESS' "
                "2>/dev/null"
            )
            result = run_command(test_cmd, check=False)
            if result.returncode == 0:
                logger.info(f"{source_machine} → {target_machine}: ✓ SUCCESS")
            else:
                logger.error(f"{source_machine} → {target_machine}: ✗ FAILED")
                all_success = False

    if all_success:
        logger.info("\n✓ Setup complete! All connections successful.")
    else:
        logger.warning("\n⚠ Setup complete with some failures. Check logs for details.")


if __name__ == "__main__":
    # Parse command line arguments
    if len(sys.argv) == 1:
        # No arguments - show usage and exit
        print("Usage: python setup_passwordless_ssh.py <username> [cluster_config.yaml]")
        print(f"\nDefault cluster config: {DEFAULT_CLUSTER_CONFIG}")
        print("\nExamples:")
        print("  python gtsfm/utils/setup_passwordless_ssh.py zliu890")
        print("  python gtsfm/utils/setup_passwordless_ssh.py zliu890 gtsfm/configs/cluster.yaml")
        sys.exit(1)
    elif len(sys.argv) == 2:
        # Only username provided - use default cluster config
        username = sys.argv[1]
        cluster_config = DEFAULT_CLUSTER_CONFIG
        logger.info(f"Using default cluster config: {cluster_config}")
    elif len(sys.argv) == 3:
        # Both username and cluster config provided
        username = sys.argv[1]
        cluster_config = sys.argv[2]
    else:
        logger.error("Too many arguments")
        print("Usage: python setup_passwordless_ssh.py <username> [cluster_config.yaml]")
        sys.exit(1)

    setup_passwordless_ssh(cluster_config, username)
