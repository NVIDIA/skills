# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Detect and reconcile native benchmark/probe logs without inferring losslessness."""
import argparse
import json
from pathlib import Path
import re


APP = re.compile(
    r"^(TX|RX) complete: (?:interface=(\S+) (queue=(\d+)|queues=(all)) )?"
    r"packets=(\d+)\b"
)
STOP = re.compile(
    r"RX worker stopped for port (\d+) queue (\d+): "
    r"cqe_seen=(\d+) data=(\d+) filler=(\d+) err=(\d+)\b"
)
ENGINE_RX = re.compile(
    r"ibverbs RX port (\d+) q(\d+) \([^)]*\): "
    r"packets=(\d+) fillers=(\d+) cqe_errors=(\d+)\b"
)
ENGINE_TX = re.compile(
    r"ibverbs TX port (\d+) q(\d+) \([^)]*\): "
    r"posted=(\d+) completed=(\d+) inflight=(\d+) "
    r"handoff_full_drops=(\d+) bursts \((\d+) pkts\)"
)
HW_REORDER = re.compile(r"ibverbs HW reorder port (\d+) q(\d+): (.*)")
COUNTER = re.compile(r"^\s*([A-Za-z0-9_]+):\s*(\d+)\s*$")
INTEGRITY = re.compile(r"^INTEGRITY_RESULT\s+(.+)$", re.M)
SELECTED_COUNTERS = (
    "vport_loopback_packets", "vport_loopback_bytes", "tx_packets_phy", "rx_packets_phy",
    "tx_bytes_phy", "rx_bytes_phy", "rx_out_of_buffer", "rx_discards_phy", "tx_discards_phy",
    "rx_errors_phy", "rx_crc_errors_phy", "rx_steer_missed_packets", "rx_pause_ctrl_phy", "tx_pause_ctrl_phy",
    "rx_vport_unicast_packets", "rx_vport_unicast_bytes",
    "tx_vport_unicast_packets", "tx_vport_unicast_bytes",
)


def counter_snapshot(text):
    if text.lstrip().startswith(("{", "[")):
        def unique_object(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError("Duplicate JSON counter key: "+key)
                result[key] = value
            return result
        data = json.loads(text, object_pairs_hook=unique_object)
        if not isinstance(data, dict):
            raise ValueError("Counter JSON must contain a numeric object, optionally scoped by host/port.")
        values = {}
        def flatten(node, prefix=""):
            for key, value in node.items():
                name = prefix+key
                if isinstance(value, dict):
                    flatten(value, name+"/")
                elif type(value) is int and value >= 0:
                    values[name] = value
                else:
                    raise ValueError("Non-counter JSON leaf: "+name)
        flatten(data)
        return values, []
    values, duplicates = {}, []
    for line in text.splitlines():
        match = COUNTER.match(line)
        if match:
            name, value = match.groups()
            if name in values:
                duplicates.append(name)
            values[name] = int(value)
    return values, duplicates


def reconcile_counters(before_text, after_text):
    try:
        before, duplicates_before = counter_snapshot(before_text)
        after, duplicates_after = counter_snapshot(after_text)
    except (ValueError, TypeError) as error:
        return {"status": "unsupported", "reason": str(error),
                "interpretation": "Inspect these snapshots directly; no missing values were treated as zero."}
    common = before.keys() & after.keys()
    resets = sorted(name for name in common if after[name] < before[name])
    duplicates = sorted(set(duplicates_before+duplicates_after))
    valid = common - set(resets) - set(duplicates)
    deltas = {name: after[name]-before[name] for name in sorted(valid)}
    selected = [name for name in sorted(deltas) if name.rsplit("/", 1)[-1] in SELECTED_COUNTERS]
    return {
        "status": "ambiguous" if resets or duplicates or not common else "parsed",
        "matched_counters": len(common),
        "selected_deltas": {name: deltas[name] for name in selected},
        "selected_levels": {name: {"before": before[name], "after": after[name], "delta": deltas[name]}
                            for name in selected},
        "unchanged_positive_selected_levels": {name: before[name] for name in selected
                                                if before[name] > 0 and deltas[name] == 0},
        "selected_missing_or_ambiguous": [name for name in SELECTED_COUNTERS
                                           if not any(key.rsplit("/", 1)[-1] == name for key in selected)],
        "all_nonzero_deltas": {name: value for name, value in deltas.items() if value},
        "reset_wrap_or_scope_change": resets,
        "duplicate_counter_names": duplicates,
        "keys_only_before": sorted(before.keys()-after.keys()),
        "keys_only_after": sorted(after.keys()-before.keys()),
        "interpretation": "Same-scope/name deltas only; aliases and ports are not added. "
                          "A positive unchanged baseline is not an absolute zero counter. "
                          "Deltas do not establish complete delivery or counter provenance.",
    }


def summarize_monitor(text):
    manual_stop = text.rstrip().endswith("KeyboardInterrupt")
    errors = [line.strip() for line in text.splitlines() if re.search(
        r"command not found|no such file|modulenotfounderror|failed|cannot execute", line, re.I)]
    if "Traceback (most recent call last):" in text and not manual_stop:
        errors.append("Unresolved monitor traceback; inspect the saved output.")
    rows = re.findall(r"^\s*([A-Za-z0-9_]+):.*?=\s*([0-9.,]+)\s+Mbps\b", text, re.M)
    return {"dependency_errors": errors, "recognized_rate_rows": len(rows),
            "termination": "KeyboardInterrupt" if manual_stop else "not_determined",
            "counters_with_rate_rows": sorted({name for name, _ in rows}),
            "measurement_status": "invalid" if errors or not rows else "samples_present_not_qualified",
            "qualification": "Stable timing, duration, counter aliases and path require review; "
                             "no sustained-throughput verdict is inferred from row count or exit status."}


def summarize_config(path):
    try:
        import yaml
    except ImportError:
        return {"status": "unparsed", "reason": "PyYAML unavailable; inspect the supplied config directly."}
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        cfg = data["daqiri"]["cfg"]
        receive = [interface["rx"] for interface in cfg.get("interfaces", []) if "rx" in interface]
        source_packets = cfg.get("stream_type") == "raw" and receive and all(
            isinstance(rx, dict) and isinstance(rx.get("queues"), list) and rx["queues"]
            and all(isinstance(queue, dict) for queue in rx["queues"])
            and rx.get("reorder_configs", []) == [] for rx in receive)
        return {"status": "parsed_not_semantically_validated", "stream_type": cfg.get("stream_type"),
                "engine": cfg.get("engine"), "loopback": cfg.get("loopback"),
                "application_rx_count_unit": "source_packets" if source_packets else "unknown",
                "memory_regions": [{key: region.get(key) for key in
                                    ["name", "kind", "num_bufs", "buf_size"]}
                                   for region in cfg.get("memory_regions", [])]}
    except (ValueError, KeyError, TypeError, AttributeError, yaml.YAMLError) as error:
        return {"status": "unparsed", "reason": str(error)}


def choose_one(paths, description, warnings):
    paths = sorted(set(path for path in paths if path.is_file()))
    if len(paths) > 1:
        warnings.append("Multiple "+description+" files; select explicit paths.")
        return None
    return paths[0] if paths else None


def detect_format(text, suffix=""):
    probe = bool(INTEGRITY.search(text))
    bench = any(APP.search(line) for line in text.splitlines())
    if probe and bench:
        return "mixed_application_formats"
    if probe:
        return "integrity_probe"
    if bench or STOP.search(text) or ENGINE_RX.search(text) or ENGINE_TX.search(text) or HW_REORDER.search(text):
        return "native_raw"
    if suffix.lower() == ".csv":
        return "structured_csv"
    if text.lstrip().startswith(("{", "[")):
        try:
            json.loads(text)
        except ValueError:
            pass
        else:
            return "structured_json"
    return "unsupported_text"


def select_logs(path):
    if path.is_file():
        return [path], None
    if not path.is_dir():
        return [], "Input does not exist; select a known file or single run directory."
    bench = path/"bench.log"
    pair = [p for p in (path/"tx.log", path/"rx.log") if p.is_file()]
    if bench.is_file() and pair:
        return [], "Multiple log layouts: select bench.log or the intended explicit log instead of combining them."
    if bench.is_file():
        return [bench], None
    if pair:
        return pair, None
    return [], "No supported logs directly in this directory. Select a known run/log; nested captures are not combined."


def unsupported_report(kind, reason, paths):
    return {"status": "unsupported", "format": kind, "reason": reason,
            "input_logs": [str(p) for p in paths],
            "next_action": "Read structured evidence directly or select the known log once; do not retry guessed paths.",
            "end_to_end_losslessness": "not_established"}


def summarize_probe(text):
    report = summarize(text)
    records = {}
    for encoded in INTEGRITY.findall(text):
        try:
            row = json.loads(encoded)
            role = row["role"]
            if role not in ("tx", "rx") or role in records:
                raise ValueError("Expected at most one result for each TX/RX role; select one capture.")
            own_count = "accepted" if role == "tx" else "received"
            for key in (own_count, "expected"):
                if key == "expected" and key not in row:
                    continue
                if type(row.get(key)) is not int or row[key] < 0:
                    raise ValueError("Missing/invalid nonnegative probe count: "+key)
            records[role] = row
        except (ValueError, KeyError, TypeError) as error:
            return {"status": "ambiguous", "format": "integrity_probe", "reason": str(error),
                    "end_to_end_losslessness": "not_established"}
    tx = records.get("tx", {}).get("accepted")
    rx = records.get("rx", {}).get("received")
    gap = tx-rx if tx is not None and rx is not None else None
    report.update(application_tx_packets=tx, application_rx_packets=rx,
                  application_rx_outputs=rx, application_rx_count_unit="source_packets",
                  tx_minus_application_rx=gap, integrity_results=records,
                  capture_pairing="Supplied file selection; matching nonce/run identity is not independently verified.")
    report["warnings"] = [w for w in report["warnings"] if not w.startswith("No recognized final application")]
    report["unavailable_scopes"] = [role for role in ("tx", "rx") if role not in records]
    if gap is not None:
        report["observations"].insert(0,
            f"TX accepted minus application RX = {gap}; this is count accounting, not a payload certificate.")
    report["observations"].append("Integrity fields and pass flags are reported finite-probe observations; "
                                  "interpret against that probe source and capture, not another run or peak throughput.")
    report["status"] = "partial" if report["warnings"] or report["unavailable_scopes"] else "parsed"
    return report


def summarize_bundle(log, before=None, after=None, monitor=None, config=None, detect_only=False):
    directory = log if log.is_dir() else log.parent
    paths, reason = select_logs(log)
    if reason:
        return unsupported_report("unselected", reason, paths)
    texts = [path.read_text(encoding="utf-8", errors="replace") for path in paths]
    formats = [detect_format(text, path.suffix) for path, text in zip(paths, texts)]
    if len(set(formats)) != 1 or formats[0] not in ("native_raw", "integrity_probe"):
        return unsupported_report("/".join(formats),
                                  "This helper supports native raw or finite-probe logs, not structured tables or mixed captures.", paths)
    kind = formats[0]
    if detect_only:
        return {"status": "parsed", "format": kind, "input_logs": [str(p) for p in paths],
                "interpretation": "Format recognized only; no measurement or success verdict."}
    text = "\n".join(texts)
    config = config or (directory/"config.used.yaml" if (directory/"config.used.yaml").is_file() else None)
    configuration = summarize_config(config) if config is not None else None
    report = summarize_probe(text) if kind == "integrity_probe" else summarize(
        text, rx_source_packets=bool(configuration and
                                    configuration.get("application_rx_count_unit") == "source_packets"))
    report["format"] = kind
    report["input_logs"] = [str(p) for p in paths]
    if report["status"] == "ambiguous":
        return report
    warnings = report["warnings"]
    if before is None and after is None:
        pairs = [(p, p.with_name(p.name.removesuffix("_before.txt")+"_after.txt"))
                 for p in directory.glob("ethtool_*_before.txt")]
        pairs = [(a, b) for a, b in pairs if b.is_file()]
        if (directory/"before.json").is_file() and (directory/"after.json").is_file():
            pairs.append((directory/"before.json", directory/"after.json"))
        if len(pairs) == 1:
            before, after = pairs[0]
        elif len(pairs) > 1:
            warnings.append("Multiple NIC snapshot pairs; select --before and --after.")
    if before is not None and after is not None:
        report["nic_counters"] = reconcile_counters(
            before.read_text(encoding="utf-8", errors="replace"),
            after.read_text(encoding="utf-8", errors="replace"))
        report["nic_counters"]["inputs"] = [str(before), str(after)]
        if report["nic_counters"]["status"] != "parsed":
            warnings.append("Counter snapshots need direct review: ambiguous/unsupported scope or values.")
    elif before is not None or after is not None:
        warnings.append("A NIC counter delta needs both before and after snapshots.")
    if monitor is None:
        monitor = choose_one(list(directory.glob("mlnx_perf*.log"))+[directory/"monitor.log"],
                             "monitor", warnings)
    if monitor is not None:
        report["monitor"] = summarize_monitor(monitor.read_text(encoding="utf-8", errors="replace"))
        report["monitor"]["input"] = str(monitor)
    if config is not None:
        report["configuration"] = configuration
        report["configuration"]["input"] = str(config)
    report["status"] = "partial" if warnings or report.get("unavailable_scopes") else "parsed"
    return report


def summarize(text, *, rx_source_packets=False):
    app_rows, worker_rows, snapshots, tx_snapshots, reorder_snapshots, errors = [], [], [], [], [], []
    for number, line in enumerate(text.splitlines(), 1):
        m = APP.search(line)
        if m:
            direction, interface, _, queue, aggregate, packets = m.groups()
            app_rows.append({"line": number, "direction": direction,
                             "interface": interface,
                             "queue": "all" if aggregate else int(queue) if queue is not None else None,
                             "packets": int(packets)})
            for field in ["bytes", "bursts", "seconds"]:
                value = re.search(r"\b"+field+r"=(\d+(?:\.\d+)?)\b", line)
                if value:
                    app_rows[-1][field] = float(value.group(1)) if field == "seconds" else int(value.group(1))
        m = STOP.search(line)
        if m:
            port, queue, cqe, data, filler, err = map(int, m.groups())
            worker_rows.append({"line": number, "port": port, "queue": queue,
                                "cqe_seen": cqe, "data_packets": data,
                                "fillers": filler, "errors": err})
        m = ENGINE_RX.search(line)
        if m:
            port, queue, packets, filler, err = map(int, m.groups())
            snapshots.append({"line": number, "port": port, "queue": queue,
                              "packets": packets, "fillers": filler,
                              "cqe_errors": err})
            dropped = re.search(r"app_ring_full_drops=(\d+) bursts \((\d+) pkts\)", line)
            if dropped:
                snapshots[-1].update(app_ring_full_dropped_bursts=int(dropped[1]),
                                     app_ring_full_dropped_packets=int(dropped[2]))
        m = ENGINE_TX.search(line)
        if m:
            port, queue, posted, completed, inflight, bursts, packets = map(int, m.groups())
            tx_snapshots.append({"line": number, "port": port, "queue": queue,
                                 "posted": posted, "completed": completed, "inflight": inflight,
                                 "handoff_full_dropped_bursts": bursts,
                                 "handoff_full_dropped_packets": packets})
        m = HW_REORDER.search(line)
        if m:
            reorder_snapshots.append({"line": number, "port": int(m[1]), "queue": int(m[2]),
                                      "reported_counters": {key: int(value) for key, value
                                                            in re.findall(r"(\w+)=(\d+)", m[3])}})
        if "NO_FREE_PACKET_BUFFERS" in line or "NO_FREE_BURST_BUFFERS" in line:
            errors.append({"line": number, "text": line.strip()})

    warnings, totals = [], {}
    for direction in ["TX", "RX"]:
        rows = [r for r in app_rows if r["direction"] == direction]
        if not rows:
            warnings.append(f"No recognized final application {direction} summary.")
            totals[direction] = None
            continue
        if any(r["interface"] is None for r in rows):
            totals[direction] = rows[0]["packets"] if len(rows) == 1 else None
            if len(rows) > 1:
                warnings.append(f"Unscoped {direction} summary overlaps other summaries; "
                                "select one run before reconciliation.")
            continue
        total = 0
        for interface in sorted({r["interface"] for r in rows}):
            group = [r for r in rows if r["interface"] == interface]
            keys = [r["queue"] for r in group]
            if len(keys) != len(set(keys)):
                warnings.append(f"Repeated {direction} summary for {interface}; "
                                "select one run before reconciliation.")
                total = None
                break
            aggregate = [r for r in group if r["queue"] == "all"]
            queues = [r for r in group if r["queue"] != "all"]
            if aggregate:
                value = aggregate[0]["packets"]
                if queues and sum(r["packets"] for r in queues) != value:
                    warnings.append(f"Conflicting aggregate/per-queue {direction} "
                                    f"counts for {interface}.")
                    total = None
                    break
            else:
                value = sum(r["packets"] for r in queues)
            total += value
        totals[direction] = total

    worker_keys = [(r["port"], r["queue"]) for r in worker_rows]
    worker_data = None
    if len(worker_keys) != len(set(worker_keys)):
        warnings.append("Repeated worker-stop summary; worker total is ambiguous.")
    elif worker_rows:
        worker_data = sum(r["data_packets"] for r in worker_rows)

    reordered = reorder_snapshots or re.search(
        r"\b(?:Hardware reorder|Reorder plan|DPDK reorder|Reorder '[^']+' stream set)\b", text, re.I)
    unscoped_rx = any(r["direction"] == "RX" and r["interface"] is None for r in app_rows)
    rx_unit = "source_packets" if rx_source_packets and not reordered and not unscoped_rx else "unknown"
    tx = totals["TX"]
    rx = totals["RX"] if rx_unit == "source_packets" else None
    if totals["RX"] is not None and rx_unit == "unknown":
        warnings.append("RX summaries may count reordered aggregates; source-packet units are unknown. "
                        "Inspect the capture and matching configuration before comparing with TX.")
    gap = tx - rx if tx is not None and rx is not None else None
    observations = []
    if gap is not None and gap > 0:
        observations.append(f"Application consumed {gap} fewer packets than TX reported; "
                            "complete delivery is unverified. Inspect discard evidence and timing "
                            "before assigning a cause.")
    elif gap is not None and gap < 0:
        observations.append("Application RX exceeds TX; investigate scope/background/duplicates.")
    elif gap == 0:
        observations.append("Final application counts match; payload identity and "
                            "compensating loss/duplicates remain unverified.")
    if errors:
        observations.append("Explicit no-free-buffer errors occurred; inspect the affected pool.")
    if any(row.get("app_ring_full_dropped_packets", 0) for row in snapshots):
        observations.append("RX app-ring discard counters are positive: software discards were "
                            "reported even if NIC drop counters are zero. Do not label the whole "
                            "gap a harmless shutdown tail.")
    if any(row["handoff_full_dropped_packets"] for row in tx_snapshots):
        observations.append("TX handoff discard counters are positive; inspect TX queue backpressure.")
    if reorder_snapshots:
        observations.append("Hardware-reorder packets, placements, batches and application output "
                            "can use different units/scopes; inspect geometry/publication semantics.")
    observations.append("Worker totals, application totals and pre-stop snapshots describe "
                        "different boundaries; do not add them together.")
    return {
        "status": "incomplete" if warnings else "parsed",
        "application_tx_packets": tx,
        "application_rx_packets": rx,
        "application_rx_outputs": totals["RX"],
        "application_rx_count_unit": rx_unit,
        "tx_minus_application_rx": gap,
        "worker_stop_data_packets": worker_data,
        "application_summaries": app_rows,
        "pre_stop_engine_rx_snapshots": snapshots,
        "pre_stop_engine_tx_snapshots": tx_snapshots,
        "hardware_reorder_snapshots": reorder_snapshots,
        "worker_stop_summaries": worker_rows,
        "buffer_errors": errors,
        "warnings": warnings,
        "observations": observations,
        "end_to_end_losslessness": "not_established_by_counts",
        "path_memory_payload_and_throughput": "require_separate_evidence",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path, help="native raw/probe log, or one run directory with bench.log or tx.log/rx.log")
    parser.add_argument("--detect-only", action="store_true", help="report recognized format without reconciling counts")
    parser.add_argument("--before", type=Path, help="explicit pre-run ethtool snapshot")
    parser.add_argument("--after", type=Path, help="explicit post-run ethtool snapshot")
    parser.add_argument("--monitor", type=Path, help="explicit monitor output")
    parser.add_argument("--config", type=Path, help="explicit configuration copy (parsing is not validation)")
    args = parser.parse_args()
    try:
        report = summarize_bundle(args.log, args.before, args.after, args.monitor, args.config, args.detect_only)
    except (OSError, UnicodeError) as error:
        report = {"status": "invalid", "reason": str(error), "end_to_end_losslessness": "not_established"}
    print(json.dumps(report, indent=2))
    # A zero status means parsing succeeded, not that the benchmark passed.
    return 0 if report["status"] == "parsed" else 3 if report["status"] == "unsupported" else 2


if __name__ == "__main__":
    raise SystemExit(main())
