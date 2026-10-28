#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Deterministic mock of the Run:ai MCP — replays the frozen demo-cluster testbed state.
Both benchmark arms call THIS (via --mcp-config), so tool data is identical every run.
Exposes the tools the audit skill uses; args are honored where the skill filters by them."""
import sys, json

# --- frozen state (demo-cluster, 2026-07-19) -------------------------------------
# per (project, node_pool): deserved / allocated / limit  (GPU)
CELLS = [
    ("node-pool-2","hpc-fill","rnd",       24,16,24),
    ("node-pool-2","starved-svc","default",16,12,16),
    ("node-pool-2","quota-hog","rnd",       2, 0,32),
    ("default","research-lab","rnd",         4,24,64),
    ("default","inference-prod","default",   8, 8, 8),
    ("default","underused","default",       12, 4,24),
    ("default","notebooks","rnd",            2, 2, 8),
    ("default","def-proj-1","default",       3, 3, 3),
    ("default","def-proj-2","default",       3, 3, 3),
    ("default","def-proj-3","default",       3, 3, 3),
]
CLUSTER_ID = "50000000-0000-4000-8000-0000000000f1"
CAPACITY = {"default":128, "node-pool-2":32}
# Pool and department totals below intentionally exceed the sum of the CELLS above —
# they include system + unlisted-project usage (node-pool-2: 28 listed + 4 system = 32,
# so the pool is genuinely full; default: 47 + 3; rnd dept: 42 + 7). Derive a pool's
# free capacity from get_cluster_physical_inventory, never by summing per-project cells.
ALLOCATED_POOL = {"default":50, "node-pool-2":32}
# pending workloads per project: (count, preemptible, reason)
PENDING = {
    "quota-hog":  (4, False, "GPU quota exceeded for non-preemptible workloads. Requested 4 devices; project quota 2 devices."),
    "starved-svc":(1, True,  "Pipelined — slot reserved, waiting for capacity to free (pool full of non-preemptible work)."),
    "hpc-fill":   (2, False, "Pending — node-pool-2 is physically full."),
}
# (department, node_pool): deserved, allocated — derived from the project CELLS above so
# department roll-ups reconcile with the per-project quota.
DEPTS = {
    ("rnd","node-pool-2"): (26,16), ("rnd","default"): (6,26),
    ("default","node-pool-2"): (16,12), ("default","default"): (29,21),
}

def project_resources(a):
    npf, pf, df = a.get("nodePoolName"), a.get("projectName"), a.get("departmentName")
    out=[]
    for pool,proj,dept,des,al,lim in CELLS:
        if npf and npf!=pool: continue
        if pf and pf!=proj: continue
        if df and df!=dept: continue
        out.append({"nodePoolName":pool,"projectName":proj,"departmentId":dept,
                    "gpu":{"deserved":des,"allocated":al,"limit":lim,"overQuotaWeight":2}})
    return {"resources":out,"hasMore":False}

class ToolError(Exception):
    pass

# One running workload per allocated cell and one row per pending workload.
WORKLOADS = []
for pool,proj,dept,des,al,lim in CELLS:
    if al>0:
        WORKLOADS.append({"name":f"{proj}-run-{pool}","phase":"Running","projectName":proj,
                          "currentNodePools":[pool],"preemptible":True,"phaseMessage":""})
for proj,(count,pre,reason) in PENDING.items():
    for i in range(count):
        WORKLOADS.append({"name":f"{proj}-pending-{i+1}","phase":"Pending","projectName":proj,
                          "currentNodePools":[],"preemptible":pre,"phaseMessage":reason})
for i,row in enumerate(WORKLOADS):
    row["id"]=f"00000000-0000-4000-8000-{i+1:012d}"
WORKLOAD_BY_ID={row["id"]:row for row in WORKLOADS}

def workloads_summary(a):
    proj=a.get("orgName"); npf=a.get("nodePoolName")
    rows=[r for r in WORKLOADS if (not proj or r["projectName"]==proj)
          and (not npf or npf in r["currentNodePools"])]
    pend=sum(1 for r in rows if r["phase"]=="Pending")
    running=sum(1 for r in rows if r["phase"]=="Running")
    return {"scope":{"orgType":a.get("orgType"),"orgName":proj,"nodePool":npf},
            "counts":{"pending":pend,"running":running,"failed":0,"completed":0,"stopped":0,"other":0},
            "total":pend+running}

def project_filter(value):
    if value is None or isinstance(value,str): return value
    if isinstance(value,dict) and value.get("operator")=="==": return value.get("value")
    raise ToolError("validation: this fixture supports only projectName with operator ==")

def list_workloads(a):
    proj=project_filter(a.get("projectName")); phase=a.get("phase"); npf=a.get("nodePoolName")
    limit=int(a.get("limit") or 50); offset=int(a.get("offset") or 0)
    rows=[r for r in WORKLOADS if (not proj or r["projectName"]==proj)
          and (not phase or r["phase"]==phase) and (not npf or npf in r["currentNodePools"])]
    page=rows[offset:offset+limit]
    out=[]
    for r in page:
        item={"id":r["id"],"name":r["name"],"type":"Training","phase":r["phase"],
              "projectName":r["projectName"],"clusterId":CLUSTER_ID}
        if a.get("includeDetails"):
            item.update(preemptible=r["preemptible"],currentNodePools=r["currentNodePools"])
        out.append(item)
    more=offset+limit<len(rows)
    result={"workloads":out,"hasMore":more}
    if more: result["nextOffset"]=offset+limit
    return result

def workload_status(a):
    wid=a.get("workloadId")
    if not wid: raise ToolError("validation: workloadId is required")
    row=WORKLOAD_BY_ID.get(wid)
    if row is None: raise ToolError(f"not_found: workload {wid} not found")
    status={"id":row["id"],"name":row["name"],"clusterId":CLUSTER_ID,"type":"Training",
            "phase":row["phase"],"projectName":row["projectName"]}
    if row["phaseMessage"]: status["phaseMessage"]=row["phaseMessage"]
    return status

def physical_inventory(a):
    npf=a.get("nodePoolName")
    return {"nodePools":[{"nodePoolName":p,"gpu":{"total":CAPACITY[p],"allocated":ALLOCATED_POOL[p],
            "free":CAPACITY[p]-ALLOCATED_POOL[p]}} for p in CAPACITY if not npf or npf==p]}

def dept_resources(a):
    df=a.get("departmentName"); npf=a.get("nodePoolName"); out=[]
    for (d,pool),(des,al) in DEPTS.items():
        if df and df!=d: continue
        if npf and npf!=pool: continue
        out.append({"departmentName":d,"nodePoolName":pool,"gpu":{"deserved":des,"allocated":al}})
    return {"resources":out}

TOOLS = {
  "list_project_resources": ("Configured GPU quota + current allocation per project per node pool.", project_resources),
  "list_department_resources": ("A department's own configured quota per node pool.", dept_resources),
  "get_workloads_summary": ("Workload counts by phase for a scope (cluster/project/department/node pool). A nodePoolName filter matches current placement, so unscheduled pending workloads drop out.", workloads_summary),
  "list_workloads": ("Workload rows with IDs, filtered by phase, projectName comparison, or current node pool. includeDetails adds preemptibility.", list_workloads),
  "get_workload_status": ("One workload's phase and phaseMessage (the scheduler's reason), by workloadId.", workload_status),
  "get_cluster_physical_inventory": ("Per-node-pool GPU capacity.", physical_inventory),
}

COMMON_PROPERTIES = {"clusterId":{"type":"string"},"nodePoolName":{"type":"string"},
    "projectName":{"type":"string"},"departmentName":{"type":"string"},
    "orgType":{"type":"string"},"orgName":{"type":"string"}}
TOOL_PROPERTIES = {
    "list_workloads": {"clusterId":{"type":"string"},"nodePoolName":{"type":"string"},"phase":{"type":"string"},
        "projectName":{"type":"object","properties":{"operator":{"type":"string"},"value":{"type":"string"}}},
        "includeDetails":{"type":"boolean"},"limit":{"type":"integer"},"offset":{"type":"integer"}},
    "get_workload_status": {"workloadId":{"type":"string"}},
}

def schema():
    return [{"name":n,"description":d,
        "inputSchema":{"type":"object","properties":TOOL_PROPERTIES.get(n,COMMON_PROPERTIES)},
        "annotations":{"readOnlyHint":True,"destructiveHint":False}}
        for n,(d,_) in TOOLS.items()]

def send(obj): sys.stdout.write(json.dumps(obj)+"\n"); sys.stdout.flush()

def main():
    for line in sys.stdin:
        line=line.strip()
        if not line: continue
        try: req=json.loads(line)
        except Exception:
            send({"jsonrpc":"2.0","id":None,"error":{"code":-32700,"message":"parse error"}})
            continue
        m=req.get("method"); rid=req.get("id")
        if m=="initialize":
            send({"jsonrpc":"2.0","id":rid,"result":{"protocolVersion":"2024-11-05",
                  "capabilities":{"tools":{}},"serverInfo":{"name":"runai","version":"mock"}}})
        elif m=="notifications/initialized":
            pass
        elif m=="tools/list":
            send({"jsonrpc":"2.0","id":rid,"result":{"tools":schema()}})
        elif m=="tools/call":
            p=req.get("params",{}); name=p.get("name"); args=p.get("arguments",{}) or {}
            fn=TOOLS.get(name)
            if not fn:
                send({"jsonrpc":"2.0","id":rid,"error":{"code":-32601,"message":f"unknown tool {name}"}})
            else:
                try:
                    res=fn[1](args)
                except (ToolError,ValueError,TypeError) as error:
                    send({"jsonrpc":"2.0","id":rid,"result":{"content":[{"type":"text","text":str(error)}],"isError":True}})
                    continue
                send({"jsonrpc":"2.0","id":rid,"result":{"content":[{"type":"text","text":json.dumps(res)}]}})
        elif rid is not None:
            send({"jsonrpc":"2.0","id":rid,"error":{"code":-32601,"message":f"unknown method {m}"}})

if __name__=="__main__": main()