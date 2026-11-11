// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
/**
 * Recorder extension (port of the DEFT kit's PostToolUse record.sh).
 *
 * Appends every substantive EXECUTED command to the run's commands.log so
 * later sessions and later runs LOAD AND RUN instead of re-deriving.
 *
 * Fidelity notes vs record.sh:
 *  - command text is captured at tool_call time (keyed by toolCallId) and
 *    written on tool_execution_end. Guard-blocked calls are suppressed two
 *    ways: (1) when guard.ts loads BEFORE this file (-e order), a block
 *    short-circuits the tool_call chain so `pending` is never set; (2) as
 *    defense in depth, tool_execution_end results that ARE a guard block
 *    (Pi emits it for blocked calls too, as an immediate error result) are
 *    detected by their GUARD( reason text and skipped. Executed-but-failed
 *    commands are still recorded, matching PostToolUse semantics.
 *  - run dir = $PI_KIT_RD if the driver exported it, else the newest
 *    $PI_KIT_RUN_PREFIX* (default run_*) dir under $PI_KIT_WS/results
 *    (bare $WS honored as fallback; mtime sort = record.sh's `ls -td | head -1`).
 */
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import * as fs from "node:fs";
import * as path from "node:path";

// Substantive-command patterns. The defaults cover the bank's shipped card
// packs (DEFT AOI + AutoML); add workflow-specific ones without editing this
// file via PI_KIT_RECORD_PATTERNS (comma-separated regex fragments).
const PATTERNS = [
	/docker run/,
	/visual_changenet/,
	/gap_analysis/,
	/nearest_neighbors/,
	/image_embeddings/,
	/analyze_kpi/,
	/log_stage/,
	/commit_stage/,
	/validate_training_csv/,
	...(process.env.PI_KIT_RECORD_PATTERNS ?? "")
		.split(",")
		.map((s) => s.trim())
		.filter(Boolean)
		.flatMap((s) => {
			// A throw here happens at extension load and would kill every session
			// (pi hard-exits on extension load errors) — degrade to a logged skip.
			try {
				return [new RegExp(s)];
			} catch (e) {
				console.error(`recorder.ts: ignoring invalid PI_KIT_RECORD_PATTERNS fragment ${JSON.stringify(s)}: ${e}`);
				return [];
			}
		}),
];

function currentRunDir(): string | undefined {
	// Trust PI_KIT_RD when set: if the dir is missing, the append below throws
	// into its catch (record dropped) — better than misfiling this run's
	// commands into a previous run's commands.log via the fallback scan.
	const rd = process.env.PI_KIT_RD;
	if (rd) return rd;
	const ws = process.env.PI_KIT_WS ?? process.env.WS;
	if (!ws) return undefined;
	const results = path.join(ws, "results");
	const prefix = process.env.PI_KIT_RUN_PREFIX ?? "run_";
	try {
		const runs = fs
			.readdirSync(results)
			.filter((d) => d.startsWith(prefix))
			.map((d) => path.join(results, d))
			.filter((p) => fs.statSync(p).isDirectory())
			.sort((a, b) => fs.statSync(b).mtimeMs - fs.statSync(a).mtimeMs);
		return runs[0];
	} catch {
		return undefined;
	}
}

export default function (pi: ExtensionAPI) {
	const pending = new Map<string, string>();

	pi.on("tool_call", async (event) => {
		if (event.toolName !== "bash") return undefined;
		const cmd = String(event.input.command ?? "");
		if (cmd && PATTERNS.some((p) => p.test(cmd))) pending.set(event.toolCallId, cmd);
		return undefined;
	});

	pi.on("tool_execution_end", async (event) => {
		const cmd = pending.get(event.toolCallId);
		if (!cmd) return;
		pending.delete(event.toolCallId);
		if (event.isError && /GUARD\(/.test(JSON.stringify(event.result ?? ""))) return; // blocked, never executed
		const rd = currentRunDir();
		if (!rd) return;
		try {
			fs.appendFileSync(path.join(rd, "commands.log"), `### ${new Date().toISOString()}\n${cmd}\n\n`);
		} catch {
			/* never break the run over logging */
		}
	});
}
