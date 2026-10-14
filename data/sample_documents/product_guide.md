# Northwind Cloud Backup Service

Northwind is a managed backup service for small and medium businesses. It backs
up servers, workstations and databases to encrypted cloud storage, and provides a
web console for monitoring and restore operations.

## Overview

Northwind stores encrypted snapshots and keeps a configurable retention window.
The service supports Windows, macOS and Linux agents. Snapshots are incremental
after the first full backup, so only changed blocks are uploaded.

## Plans and Limits

| Plan | Storage | Retention | Monthly Price |
| --- | --- | --- | --- |
| Basic | 100 GB | 30 days | $5 |
| Standard | 1 TB | 90 days | $15 |
| Enterprise | 10 TB | 365 days | $49 |

The maximum size of a single file that can be backed up is 50 GB on the Basic and
Standard plans. Enterprise plans raise the per-file limit to 500 GB.

## Installation

1. Download the Northwind agent from the customer portal.
2. Run the installer and accept the license agreement.
3. Enter your account token when prompted.
4. Select the folders you want to protect.
5. Confirm the backup schedule and finish the wizard.

## Backup Schedule

By default, Northwind runs a full backup at 02:00 UTC every Sunday and an
incremental backup daily at 02:00 UTC. Schedules can be changed per agent.

## Restore Procedure

1. Open the Northwind console and locate the source machine.
2. Choose a recovery point from the snapshot timeline.
3. Select the files or folders to restore.
4. Restore to the original location or to an alternate path.
5. Verify the restored files before deleting the recovery point.

## Security and Encryption

All data is encrypted at rest with AES-256 and in transit with TLS 1.3. Encryption
keys are stored in a hardware security module. Northwind staff cannot read
customer backup data.

## Warnings and Exceptions

- Warning: deleting an agent does not delete its stored snapshots automatically.
- Warning: backups older than the retention window are removed permanently.
- Exception: the Basic plan cannot be used for database backups.
- Exception: files larger than the plan limit are skipped and reported as errors.
