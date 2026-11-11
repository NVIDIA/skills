// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
/**
 * NVIDIA Inference API provider for Pi.
 *
 * Registers the internal OpenAI-compatible endpoint as provider "nim" so the
 * kit can run on Nemotron instead of Anthropic:
 *
 *   pi --model "nim/nvidia/nvidia/Nemotron-3-Nano-30B-A3B" ...
 *
 * The API key is NEVER stored in this repo: it resolves from the
 * NVIDIA_INFERENCE_API_KEY environment variable at request time.
 *
 * Endpoint facts (verified 2026-07-22 via curl):
 *  - /v1/chat/completions, Bearer auth, model field must be the full
 *    "nvidia/nvidia/Nemotron-3-Nano-30B-A3B" string
 *  - tool calling works (finish_reason "tool_calls")
 *  - responses carry a DeepSeek-style `reasoning_content` field (thinking is
 *    on server-side by default); usage reports prompt/completion tokens only
 *    (no cache accounting)
 */
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

const NIM_MODEL_PREFIX = "nvidia/";   // covers nvidia/nvidia/* and nvidia/qwen/*

export default function (pi: ExtensionAPI) {
	// Greedy decoding for executor work: at default sampling the nano model
	// sometimes SIMULATES tool output in prose instead of calling the tool
	// (observed ~50% on trivial prompts). temperature 0 pins it to the
	// tool-calling path. Applied to every model on this provider.
	pi.on("before_provider_request", (event) => {
		const p = event.payload as Record<string, unknown> | null;
		if (p && typeof p === "object" && typeof p.model === "string" && p.model.startsWith(NIM_MODEL_PREFIX)) {
			return { ...p, temperature: 0 };
		}
		return undefined;
	});

	pi.registerProvider("nim", {
		name: "NVIDIA Inference API",
		baseUrl: "https://inference-api.nvidia.com/v1",
		apiKey: "$NVIDIA_INFERENCE_API_KEY",
		api: "openai-completions",
		models: [
			{
				id: "nvidia/nvidia/Nemotron-3-Nano-30B-A3B",
				name: "Nemotron 3 Nano 30B A3B",
				// reasoning MUST be true for pi to emit chat_template_kwargs at all
				// (pi-ai openai-completions.js:518). Thinking is then controlled by
				// the model-ref suffix: `nim/...:off` -> enable_thinking:false.
				// The endpoint's server-side default is thinking ON, which burns the
				// whole completion budget on reasoning (measured 29.9k chars, died at
				// the max_tokens cap without a tool call) — always pass :off for
				// card execution.
				reasoning: true,
				input: ["text"],
				cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
				contextWindow: 131072,
				maxTokens: 16384,
				compat: {
					supportsDeveloperRole: false,
					supportsReasoningEffort: false,
					maxTokensField: "max_tokens",
					supportsUsageInStreaming: true,
					thinkingFormat: "chat-template",
					chatTemplateKwargs: { enable_thinking: { $var: "thinking.enabled" } },
				},
			},
			{
				id: "nvidia/qwen/qwen3.6-35b-a3b",
				name: "Qwen 3.6 35B A3B",
				// Same endpoint behavior (verified 2026-07-23): thinking on by
				// default, disabled via chat_template_kwargs (native Qwen
				// convention); tool calls OK thinking-off; 16384 accepted.
				reasoning: true,
				input: ["text"],
				cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
				contextWindow: 131072,
				maxTokens: 16384,
				compat: {
					supportsDeveloperRole: false,
					supportsReasoningEffort: false,
					maxTokensField: "max_tokens",
					supportsUsageInStreaming: true,
					thinkingFormat: "chat-template",
					chatTemplateKwargs: { enable_thinking: { $var: "thinking.enabled" } },
				},
			},
			{
				id: "nvidia/qwen/qwen3-5-397b-a17b",
				name: "Qwen 3.5 397B A17B",
				// Served on the PROD endpoint (inference-api.nvidia.com) — the -dev
				// endpoint rejected the prod key; verified 2026-07-23 the prod key
				// accesses this id directly. Same thinking-off + tool-call behavior.
				reasoning: true,
				input: ["text"],
				cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
				contextWindow: 131072,
				maxTokens: 16384,
				compat: {
					supportsDeveloperRole: false,
					supportsReasoningEffort: false,
					maxTokensField: "max_tokens",
					supportsUsageInStreaming: true,
					thinkingFormat: "chat-template",
					chatTemplateKwargs: { enable_thinking: { $var: "thinking.enabled" } },
				},
			},
			{
				id: "nvidia/nvidia/nemotron-3-super-v3",
				name: "Nemotron 3 Super v3",
				// Same endpoint behavior as Nano (verified 2026-07-22): thinking on
				// by default, disabled via chat_template_kwargs; tool calls OK with
				// thinking off; max_tokens 16384 accepted.
				reasoning: true,
				input: ["text"],
				cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
				contextWindow: 131072,
				maxTokens: 16384,
				compat: {
					supportsDeveloperRole: false,
					supportsReasoningEffort: false,
					maxTokensField: "max_tokens",
					supportsUsageInStreaming: true,
					thinkingFormat: "chat-template",
					chatTemplateKwargs: { enable_thinking: { $var: "thinking.enabled" } },
				},
			},
		],
	});
}
