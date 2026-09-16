---
name: rtx-remix-modding
description: Mod or remaster a game with RTX Remix - open and edit projects, swap textures and models. Connect to the Remix Toolkit App via MCP. Not for non-Remix game interaction.
metadata:
  author: scfitzpatric <scfitzpatric@nvidia.com>
  tags:
    - rtx-remix
    - game-modding
    - usd
    - mcp
    - graphics
---

# RTX Remix Modding

## When to Use This Skill

Use this when the user is modding or remastering a classic DirectX 8/9 game with NVIDIA RTX
Remix and a Remix MCP server is available. It covers opening and closing projects, working with
the project's USD layers, swapping the model and texture assets bound to captured prims, and
reading the stage. It does not cover installing Remix, capturing a game, or authoring new
geometry.

## Connecting to Remix

Requires RTX Remix `1.6.0+` or `1.6.0-dev`; ask the user to upgrade if older.

Missing `remix_*` tools? Ask the user to start Toolkit, windowed or via
`lightspeed.app.trex.stagecraft.headless.bat`, and connect to `http://127.0.0.1:18014/mcp/`.
Keep the trailing slash. Both modes fall back within 18014–18019; all busy means startup fails.

Windows discovery record, published by Remix:
`%LOCALAPPDATA%\NVIDIA\RTX Remix\mcp.json`.
Read `mcp_endpoint` for the latest instance. Contains addresses, ports, version, and PID;
removed on clean shutdown.

No record? Use `endpoint=` from the Toolkit log's `MCP_PORT_FALLBACK` warning.
Wait for matching `SERVICE_READY service=mcp`. Client URLs need manual updating.

## Instructions

Before using the MCP tools to replace models or add an asset reference, read
`references/tool-mode.md` and follow the relevant recipe.

Your tool list already carries every tool and its schema. What it does not carry is the order:

1. Open or confirm the project.
2. **Set the edit target before writing.** A write lands in whichever layer is current, and the
   wrong layer is the failure that looks like success.
3. Find the prims — `selection=true` acts on the viewport selection.
4. Write. A file has to be ingested before it can be referenced.
5. `remix_save_layer`, then read back what you changed.

## Troubleshooting

Before planning, check for `remix_*` tools; if absent, say the server is unavailable and relay
the setup instructions in "Connecting to Remix".

Never substitute a workaround for the missing tools. Editing `.usda` files by hand or searching
the filesystem for textures goes around the command layer, and the result usually lands in a
layer the runtime ignores — which looks like success and renders nothing.

When a call fails, read the response before trying again. A missing project, an unopened stage,
or a rejected path is a precondition to fix — repeating the call only repeats the failure. Retry
only what failed for a transient reason, such as a dropped connection.

## Rules

Scope caveat you must always disclose: edits apply per captured asset, not per instance, so every instance sharing that mesh changes too. This is expected and fine, but say it in your closing sentence whenever the subject has siblings (e.g. "Swapped that door's model — it shares an asset with 5 other doors in this project, so they changed too.").

Use `remix_get_model_instances` to identify a model's shared instances when needed; do not infer siblings from matching replacement filenames or invent a count.

There is no undo tool in this release. Never simulate one by hand either — any route you take to put something back yourself is the wrong one. That includes stripping your own edits back out of a layer, even after confirming they are only yours. So does authoring `references = None` or emptying a replacement prim: a blocked reference composes to nothing, deleting the asset from the game with no error anywhere. If the user wants a change reverted, say that you cannot revert it and let them do it in the Toolkit.

What you can and cannot do, by thing. Nothing outside this exists — no tool and no workaround.
Say so, and offer the nearest thing you can do:

- **Layers** — the only thing you can create or delete.
- **Projects** — open and close an existing one. You cannot create one.
- **Prims** — read them and set the selection. You cannot create, delete, rename, duplicate or
  move one. That covers geometry of every kind: no meshes, no new objects, by any route.
- **Asset references** — add one to a prim, or replace one already there. There is no removal.
- **Textures** — read what a material binds, and replace it.

Nothing saves itself in this release. A write lands in memory and the running game keeps rendering the old asset until the layer is on disk, so a turn that ends on a successful edit has changed nothing the user can see. Call `remix_save_layer` once the edit is in, then read back to confirm it landed. This is the same silent success as above, and the easier one to report as a win.

Ask instead of guessing scope. "Swap all the broken ones", "fix everything in this layer" — ask how many, and which, before acting.

Stop repeating a failing approach. Long tasks take many rounds; keep going while each round gets closer — a new error is progress. But when the same call fails the same way about three times, stop and report the error.

A call that succeeded and told you nothing useful is the same dead end, and it is the easier one to miss because nothing looks broken. **NEVER re-run a query you have not invalidated.** If nothing has happened since it last ran, it will return exactly what it returned, and running it a third time spends your turn budget on an answer you already hold. This covers searching the stage as much as anything else — a name that was not there is still not there. When a search comes back empty, that emptiness IS your answer: report what you looked for, report what you DID find, and ask the user to name the thing another way. Ending your turn with a question is a correct outcome, not a failure, and it beats spending every remaining call re-asking something you have already answered.

A tool saying the thing is not there is the WHOLE answer, not the first of two opinions. Those tools read the same stage you would walk yourself, so asking a second one returns the same nothing — and hunting through prim children for a texture the material does not carry, or a name the capture does not hold, is the loop, not the way out of it. Plenty of what a game renders is not in a capture at all: sky, fog, water, shadows, the HUD. Say what you found and that you cannot reach it. One thing DOES invalidate a query, and re-running is right for it: **you changed the scene** — read back after a write to confirm it landed.

If no tool can do what the user asked, say so and stop. "I can't do that with the tools I have — here is what I can do instead" is a correct, useful answer. Improvising around a missing capability is how real damage happens; a plain refusal costs the user one turn and nothing else.

When a tool refuses, RELAY WHAT IT TOLD YOU. These refusals are written for the user, not for you: they name the specific thing that blocked the call and, where one exists, the remedy. Pass both on verbatim in your own sentence. Two failures to avoid — dropping the remedy, so the user hears "impossible" when the tool said "do X and this works"; and explaining the CAUSE yourself when the tool did not give you one. If you find yourself writing "likely" or "probably" about why something is blocked, you are guessing at the user's setup. Say what the tool reported and what would unblock it, and stop there.

When you have finished the user's request, end your turn with one short, plain-language sentence telling the user what you did or answering their question — for an action, confirm it (e.g. "Opened your most recent project.").

Short never means contentless. "Done." and "OK" are not acceptable answers — they tell the user nothing and hide whether the thing they asked for actually happened. Name what changed, and if any rule above told you to state a caveat or a refusal, that caveat *is* the sentence. Say it even when it is the only thing you have to report.
