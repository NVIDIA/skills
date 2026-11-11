# NFS and RWX remediation

## Prefer existing storage

The default chart references StorageClass `nfs` for PostgreSQL, Airflow logs, and the model-cache
PVC. If the cluster already provides another ReadWriteMany class, prefer it and update all three
locations in `deploy/values.yaml` consistently:

- `postgresql.primary.persistence.storageClass`
- `airflow.logs.persistence.storageClassName`
- `global.modelCache.storageClassName`

Changing a StorageClass does not prove it supports ReadWriteMany. Require provider documentation,
an existing bound RWX PVC, or an explicitly approved disposable PVC/mount probe.

## Gate the repository installer

`make install nfs` uses a privileged Job to chroot into a selected node, install
`nfs-kernel-server` and `rpcbind`, write `/etc/exports`, enable and restart host services, deploy
the NFS subdir provisioner, and create a StorageClass. Use it only when all of these are true:

- The cluster is self-managed and the chosen node is compatible with the Ubuntu-based `apt` and
  `systemctl` procedure.
- The user explicitly chooses `NFS_NODE_HOSTNAME` and confirms its `InternalIP` as `NFS_SERVER`;
  do not accept the installer's first-node default silently.
- Privileged pods, host root mounts, host PID/network, cluster-scoped RBAC, package installation,
  and NFS mounts are allowed.
- `envsubst` is installed on the workstation.
- The node has enough durable disk for configured PVC requests. Current defaults include 500Gi
  model cache, 20Gi Airflow logs, and 10Gi PostgreSQL, plus headroom.
- No active workload is using NFS from that host. Restarting NFS underneath mounted workloads can
  hang I/O or the node.

Managed services, immutable nodes, non-Debian hosts, or restrictive admission policies fail this
gate. Ask the cluster administrator for a supported RWX provisioner instead.

## Approval and verification

Before execution, show the resolved node, IP, export path, namespace, StorageClass, and exact
command, for example:

```bash
make install nfs \
  NFS_NODE_HOSTNAME=<approved-node> \
  NFS_SERVER=<approved-internal-ip>
```

Run only after approval. Stop on the first failure. Then verify the setup Job completed, the
`nfs-subdir-external-provisioner` Deployment is Available, the StorageClass exists with the
expected provisioner, and recent events include nothing of type `Warning`. A PVC/mount test is a separate cluster
mutation and needs its own approval; clean up only the resources created for that test.
