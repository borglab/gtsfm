# PostgreSQL Integration for Distributed Two-View Estimation

## Key Features
- **Local/Remote Computing**: Local Dask scheduler with remote workers connected via SSH tunnels
- **Database Integration**: PostgreSQL from Docker storage for two-view estimation results and detailed reports

## Setup and Usage

- PostgreSQL Database: Ensure PostgreSQL is set up according to the *.yaml and running locally before GTSFM
- Remote Machines: Ensure all remote worker machines are on the same branch
- SSH Access: Configure passwordless SSH access to remote machines


## Architecture

[Local Machine]
├── Dask Scheduler port 8788, dashboard 8787
├── PostgreSQL Database (port 5432)
└── SSH Tunnels to Remote Workers
[Remote Workers]
├── Dask Workers port 9000
├── SSH Tunnel Connection
└── Database Access via Tunnel

### Configuration
Edit `gtsfm/configs/local_scheduler_postgres_remote_cluster.yaml`:
```yaml
username: your_username
workers:
  - your-remote-server.com
database:
  host: localhost
  port: 5432
  database: postgres
  user: postgres
  password: "your_password"
```

### Running Tests
```bash
# Basic cluster functionality test
python scripts/test_local_scheduler_db_remote_worker_config.py

# Full two-view estimator test
python scripts/test_two_view_estimator_dask_postgres.py
```