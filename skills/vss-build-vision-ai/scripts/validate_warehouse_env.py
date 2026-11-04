#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate warehouse Foundation constraints that `resolved.yml` cannot express.

`validate_resolved_yml.py` checks the resolved Compose model. The constraints
here are env-level: MODE, BP_PROFILE, HARDWARE_PROFILE and SAMPLE_VIDEO_DATASET
appear in no service `environment:` block, so they are structurally invisible
there. Each rule below fails at bring-up or silently at runtime -- never at
`docker compose config` time.

Env files are parsed directly, never sourced through a shell: the warehouse
`.env` carries an unquoted JSON value that shell quote-removal mangles, and the
shell environment outranks --env-file in Compose interpolation.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path


# (dataset -> expected NUM_STREAMS). Dataset and mode are independent: every
# shipped dataset now carries calibration for 2d, 3d and mv3dt, and
# overrides.env states it outright ("so dataset and mode can be chosen
# independently"). blueprint_config.yml agrees -- its only dataset rule is that
# NUM_STREAMS matches the camera count. Calibration presence is still checked
# per (dataset, mode) further down, which is what actually constrains a pairing.
DATASETS = {
    "nv-warehouse-4cams": 4,
    "warehouse-loading-dock-3cams-synthetic": 3,
    "warehouse-4cams-20mx20m-synthetic": 4,
}

# (dataset -> DATASET_TYPE). MODE=3d only: blueprint_config.yml scopes every
# DATASET_TYPE block under `3d:`, so the knob is inert for 2d, mv3dt and
# auto-calibration and must not be judged there. It is not derived from
# SAMPLE_VIDEO_DATASET in config -- the two are set independently -- so a
# mismatch is silent: the configurator renders a coherent Sparse4D bundle for
# the *other* family, and the labels differ only in confidence thresholds
# (real 0.85/0.75 vs synthetic 0.5/0.3). Custom footage carries no inferable
# provenance, so it is asked for at intake (SKILL.md Q2w-datatype) and only
# checked for validity here.
DATASET_TYPES = {"real", "synthetic"}
DATASET_FAMILY = {
    "nv-warehouse-4cams": "real",
    "warehouse-loading-dock-3cams-synthetic": "synthetic",
    "warehouse-4cams-20mx20m-synthetic": "synthetic",
}

MODES = {"2d", "3d", "mv3dt", "auto-calibration"}
BP_PROFILES = {"bp_wh", "bp_wh_kafka", "bp_wh_redis", "bp_wh_auto_calib"}

# The service lists this skill supports, keyed by the (MODE, BP_PROFILE) pair
# that selects them. overrides.env defines others; selecting one of those is a
# routing error, so the check is an allowlist rather than a denylist -- a list
# added upstream is rejected until it is reviewed here.
#
# The pair is load-bearing, not decoration. FOUNDATION_VARIANT records which
# baseline the build expanded, while MODE/BP_PROFILE/STREAM_TYPE drive the
# runtime wiring; nothing downstream reconciles them. BP_PROFILE=bp_wh_kafka
# against the exact COMPOSE_PROFILES_WH_REDIS_2D list resolves cleanly and boots
# healthy with no kafka service at all, because the redis lists carry no kafka
# key -- so the mismatch has to fail here.
VARIANT_MATRIX = {
    ("2d", "bp_wh"): {"COMPOSE_PROFILES_WH_2D"},
    ("2d", "bp_wh_kafka"): {
        "COMPOSE_PROFILES_WH_KAFKA_2D",
        "COMPOSE_PROFILES_WH_KAFKA_2D_MINIMAL",
    },
    ("2d", "bp_wh_redis"): {
        "COMPOSE_PROFILES_WH_REDIS_2D",
        "COMPOSE_PROFILES_WH_REDIS_2D_MINIMAL",
    },
    ("3d", "bp_wh_kafka"): {
        "COMPOSE_PROFILES_WH_KAFKA_3D",
        "COMPOSE_PROFILES_WH_KAFKA_3D_MINIMAL",
    },
    ("3d", "bp_wh_redis"): {
        "COMPOSE_PROFILES_WH_REDIS_3D",
        "COMPOSE_PROFILES_WH_REDIS_3D_MINIMAL",
    },
    ("mv3dt", "bp_wh_kafka"): {
        "COMPOSE_PROFILES_WH_KAFKA_MV3DT",
        "COMPOSE_PROFILES_WH_KAFKA_MV3DT_MINIMAL",
    },
    ("mv3dt", "bp_wh_redis"): {
        "COMPOSE_PROFILES_WH_REDIS_MV3DT",
        "COMPOSE_PROFILES_WH_REDIS_MV3DT_MINIMAL",
    },
    # auto-calibration is its own MODE, not a bp_profile layered onto 2d/3d/mv3dt:
    # it produces a calibration instead of consuming a shipped one, and there is
    # exactly one list for it.
    ("auto-calibration", "bp_wh_auto_calib"): {"COMPOSE_PROFILES_WH_AUTO_CALIB"},
}

IN_SCOPE_VARIANTS = {v for variants in VARIANT_MATRIX.values() for v in variants}

# Services every warehouse list carries that no capability names. The
# forward-closure prune in composition.md must not remove them.
INFRA_FLOOR = [
    "centralizedb",
    "vst-ingress",
    "sensor-bp-wait-bp-configurator",
    "turnserver-init",
    "turnserver",
    "redis",
]

# The SDRC chain rides on top of the common floor for the analytics modes.
# MODE=auto-calibration deliberately ships without it -- overrides.env: "2d/3d/mv3dt
# warehouse profiles use SDRC. MODE=auto-calibration has no sdr-controller" -- so
# requiring these of the auto-calibration list would reject a correct build.
SDRC_FLOOR = [
    "init-dirs",
    "render-config",
    "wdm-env-from-config",
    "wait-for-redis",
    "sdr-controller",
]

ASSIGN = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=(.*)$")

# Warehouse service keys carry the deployment mode as a token, e.g. perception-3d,
# bp-configurator-2d, vss-behavior-analytics-3d. The token must agree with MODE:
# a 2D detector in a 3D build resolves cleanly, boots healthy, and publishes to the
# wrong topic -- 2D writes mdx-raw while vss-behavior-analytics-3d reads mdx-bev, so
# analytics silently sees nothing.
MODE_TOKEN = re.compile(r"(?:^|-)(2d|3d|mv3dt|auto-calibration)(?:-|$)")


def strip_value(value: str) -> str:
    """Unquote and drop an inline comment, matching Compose's env_file parser.

    Compose ends an *unquoted* value at the first whitespace-preceded `#`, and
    for a quoted value takes the quoted span and ignores the remainder. Keeping
    the comment instead corrupts real values: warehouse-operations/.env ships
    `NVSTREAMER_IP=vss-vios-nvstreamer # Compose service DNS name; ...`, and a
    hostname with prose appended fails DNS inside the container.
    """
    if value[:1] in ("'", '"'):
        quote = value[0]
        end = value.find(quote, 1)
        if end != -1:
            return value[1:end]
        return value[1:]
    head, hash_sep, _ = value.partition("#")
    if hash_sep and (not head or head[-1].isspace()):
        value = head
    return value.strip()


REF_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def split_ref(inner: str) -> tuple[str, str, str]:
    """Split a `${...}` body into (name, separator, default)."""
    match = REF_NAME.match(inner)
    if not match:
        return inner, "", ""
    name = match.group(0)
    rest = inner[len(name):]
    for sep in (":-", ":?", "-"):
        if rest.startswith(sep):
            return name, sep, rest[len(sep):]
    return name, "", ""


def expand_value(value: str, lookup, _depth: int = 0) -> str:
    """Expand `${VAR}` / `${VAR:-default}` / `${VAR-default}`, defaults nesting.

    `containers.env` layers tags through nested fallbacks, e.g.
    `VSS_RT_CV_TAG="${VSS_RT_CV_TAG:-${VSS_CONTAINER_TAG:-3.3.0-26.07.2}}"`.
    Leaving that literal makes a value-level check (such as an SBSA hardware
    rule) test the fallback *expression* instead of the tag Compose resolves, so
    a build that selects an SBSA image through `VSS_CONTAINER_TAG` is rejected.

    A reference that cannot be resolved is preserved verbatim so callers can tell
    "resolved to something" from "still unknown". `$${...}` is a container-shell
    escape, not Compose interpolation, and is left alone.
    """
    if _depth > 16 or "${" not in value:
        return value
    out: list[str] = []
    i, n = 0, len(value)
    while i < n:
        if value.startswith("$${", i):
            out.append("$${")
            i += 3
            continue
        if value.startswith("${", i):
            depth, j = 1, i + 2
            while j < n and depth:
                if value[j] == "{":
                    depth += 1
                elif value[j] == "}":
                    depth -= 1
                j += 1
            if depth:  # unbalanced -- keep the remainder literal
                out.append(value[i:])
                break
            name, sep, default = split_ref(value[i + 2:j - 1])
            got = lookup(name)
            if sep == ":-":
                use_default = not got
            elif sep == "-":
                use_default = got is None
            else:
                use_default = False
            if use_default:
                out.append(expand_value(default, lookup, _depth + 1))
            elif got is not None:
                out.append(expand_value(got, lookup, _depth + 1))
            else:
                out.append(value[i:j])  # unset and no default -- leave verbatim
            i = j
            continue
        out.append(value[i])
        i += 1
    return "".join(out)


def parse_env_pairs(path: Path) -> list[tuple[str, str]]:
    """Minimal dotenv parse, in file order. No shell. Interpolation is deferred."""
    pairs: list[tuple[str, str]] = []
    if not path.is_file():
        return pairs
    for raw in path.read_text(errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        match = ASSIGN.match(line)
        if not match:
            continue
        pairs.append((match.group(1), strip_value(match.group(2).strip())))
    return pairs


def parse_env(path: Path) -> dict[str, str]:
    """Last assignment wins, values unexpanded."""
    return dict(parse_env_pairs(path))


def layered_env(repo: Path, foundation_dir: Path, build_dir: Path) -> dict[str, str]:
    """Merge the four env layers, expanding each value as the layer is read.

    Compose expands an env file as it reads it, so a later layer's `${X:-...}`
    sees the earlier layer's `X` and a self-referential default such as
    `VSS_CONTAINER_TAG="${VSS_CONTAINER_TAG:-develop-latest}"` terminates on the
    literal. The real shell environment outranks `--env-file`, so it is consulted
    before the fallback.
    """
    merged: dict[str, str] = {}

    def lookup(name: str) -> str | None:
        # Shell environment outranks --env-file in Compose interpolation, so a
        # `${VSS_CONTAINER_TAG:-...}` fallback picks up an exported tag ahead of
        # the value an earlier layer wrote.
        if name in os.environ:
            return os.environ[name]
        return merged.get(name)

    for path in (
        repo / "deploy/docker/containers.env",
        foundation_dir / ".env",
        foundation_dir / "overrides.env",
        build_dir / "override.env",
    ):
        for key, value in parse_env_pairs(path):
            merged[key] = expand_value(value, lookup)
    return merged


def check(env: dict[str, str], repo: Path, foundation_dir: Path) -> list[str]:
    errors: list[str] = []
    warnings: list[str] = []

    mode = env.get("MODE", "")
    bp = env.get("BP_PROFILE", "")
    hw = env.get("HARDWARE_PROFILE", "")
    profiles = [p for p in env.get("COMPOSE_PROFILES", "").split(",") if p]
    variant = env.get("FOUNDATION_VARIANT", "")


    if mode not in MODES:
        errors.append(
            f"MODE={mode!r} is not supported by this skill; use one of {sorted(MODES)}"
        )

    if bp not in BP_PROFILES:
        errors.append(
            f"BP_PROFILE={bp!r} is not supported by this skill; "
            f"use one of {sorted(BP_PROFILES)}"
        )

    if variant and variant not in IN_SCOPE_VARIANTS:
        errors.append(
            f"FOUNDATION_VARIANT={variant!r} is not a service list this skill "
            f"supports; use one of {sorted(IN_SCOPE_VARIANTS)}"
        )
    elif variant and mode in MODES and bp in BP_PROFILES:
        # In-scope alone is not enough: the variant must be the one this
        # (MODE, BP_PROFILE) pair selects. A mismatch deploys a coherent list
        # for the *other* pair -- every container healthy, the broker the
        # runtime expects absent.
        allowed = VARIANT_MATRIX.get((mode, bp), set())
        if variant not in allowed:
            errors.append(
                f"FOUNDATION_VARIANT={variant!r} does not match MODE={mode} + "
                f"BP_PROFILE={bp}, which selects "
                f"{sorted(allowed) if allowed else 'no supported list'}. "
                "The mismatched list resolves and boots healthy while wiring a "
                "different broker than the runtime expects"
            )

    # 1/3. bp_wh is 2d-only, and bp_wh_auto_calib is auto-calibration-only.
    if bp == "bp_wh" and mode and mode != "2d":
        errors.append(
            f"BP_PROFILE=bp_wh is unsupported with MODE={mode} "
            "(agents run in 2d only); use bp_wh_kafka or bp_wh_redis"
        )
    if bp == "bp_wh_auto_calib" and mode and mode != "auto-calibration":
        errors.append(
            f"BP_PROFILE=bp_wh_auto_calib requires MODE=auto-calibration, got MODE={mode}. "
            "There is one auto-calibration list (COMPOSE_PROFILES_WH_AUTO_CALIB); it is not "
            "a variant of the 2d/3d/mv3dt lists"
        )
    if mode == "auto-calibration" and bp and bp != "bp_wh_auto_calib":
        errors.append(
            f"MODE=auto-calibration requires BP_PROFILE=bp_wh_auto_calib, got {bp!r}"
        )

    # 2. bp_wh 2d is rejected on edge platforms.
    if bp == "bp_wh" and mode == "2d" and hw in {"IGX-THOR", "DGX-SPARK"}:
        errors.append(
            f"BP_PROFILE=bp_wh in 2d mode is not supported on {hw} "
            "(blueprint_config.yml rejects it)"
        )

    # 4. GB300 and DGX-SPARK need an sbsa perception image. The tag is layered
    #    through nested fallbacks in containers.env, so judge the expanded value;
    #    if a reference survived expansion the tag is genuinely indeterminate here
    #    and a hard error would block a build Compose resolves correctly.
    if hw in {"GB300", "DGX-SPARK"}:
        rt_cv_tag = env.get("VSS_RT_CV_TAG", "")
        if "${" in rt_cv_tag:
            warnings.append(
                f"HARDWARE_PROFILE={hw} requires an 'sbsa' perception image, but "
                f"VSS_RT_CV_TAG could not be resolved here (got {rt_cv_tag!r}); "
                "confirm the resolved tag in resolved.yml"
            )
        elif "sbsa" not in rt_cv_tag:
            errors.append(
                f"HARDWARE_PROFILE={hw} requires VSS_RT_CV_TAG to contain 'sbsa'; "
                f"got {rt_cv_tag!r}"
            )

    # 5. Dataset <-> variant <-> stream count.
    dataset = env.get("SAMPLE_VIDEO_DATASET", "")
    if dataset:
        if dataset not in DATASETS:
            warnings.append(f"SAMPLE_VIDEO_DATASET={dataset!r} is not a known sample dataset")
        else:
            streams = DATASETS[dataset]
            actual = env.get("NUM_STREAMS", "")
            if actual and actual != str(streams):
                errors.append(
                    f"NUM_STREAMS={actual} does not match dataset {dataset!r} "
                    f"(expects {streams}); a short count looks like healthy containers "
                    "processing nothing"
                )

    # 5b. DATASET_TYPE selects the Sparse4D bundle. MODE=3d only.
    if mode == "3d":
        dataset_type = env.get("DATASET_TYPE", "")
        if not dataset_type:
            # The bare consumer is streamprocessing-ms-3d, in
            # services/vios/streamprocessing/docker-compose.yaml.
            errors.append(
                "DATASET_TYPE is empty or unset under MODE=3d; it selects the "
                "Sparse4D model, anchor and labels, and the label bind source "
                "interpolates it into a host path, so an empty value mounts "
                "labels-.txt as a directory"
            )
        elif dataset_type not in DATASET_TYPES:
            errors.append(
                f"DATASET_TYPE={dataset_type!r} is not valid; "
                f"use one of {sorted(DATASET_TYPES)}"
            )
        else:
            expected_type = DATASET_FAMILY.get(dataset)
            if expected_type is None:
                # Custom dataset: provenance is not inferable, so a valid value
                # is all that can be checked. Q2w-datatype asks the user.
                pass
            elif dataset_type != expected_type:
                errors.append(
                    f"DATASET_TYPE={dataset_type!r} does not match dataset "
                    f"{dataset!r} (expects {expected_type!r}); the wrong Sparse4D "
                    "weights and detection thresholds load with every container "
                    "healthy and no error anywhere"
                )

    # 6. Broker selection must agree with the variant.
    stream_type = env.get("STREAM_TYPE", "")
    if bp == "bp_wh_redis" and stream_type != "redis":
        errors.append(f"BP_PROFILE=bp_wh_redis requires STREAM_TYPE=redis, got {stream_type!r}")
    # bp_wh_auto_calib is deliberately absent: its list carries no kafka key and no
    # perception, so it brokers no metadata and STREAM_TYPE does not apply.
    if bp in {"bp_wh", "bp_wh_kafka"} and stream_type not in {"", "kafka"}:
        errors.append(f"BP_PROFILE={bp} requires STREAM_TYPE=kafka, got {stream_type!r}")

    # 6b. The selected broker must actually be in the service list. The env
    # knobs above only state intent; COMPOSE_PROFILES decides what runs. A
    # kafka-brokered build whose list carries no `kafka` key starts clean and
    # drops every metadata record on the floor.
    if profiles and bp in {"bp_wh", "bp_wh_kafka"} and "kafka" not in profiles:
        errors.append(
            f"BP_PROFILE={bp} brokers metadata over Kafka, but COMPOSE_PROFILES "
            "contains no 'kafka' key. Perception publishes to a broker that was "
            "never deployed: containers stay healthy and no metadata is delivered"
        )
    if profiles and bp == "bp_wh_redis" and "kafka" in profiles:
        warnings.append(
            "BP_PROFILE=bp_wh_redis brokers metadata over Redis, but "
            "COMPOSE_PROFILES contains 'kafka'. No redis baseline carries a "
            "kafka key, so the list does not match the selected variant"
        )

    # 7. Local LLM needs a sizing file for this hardware profile.
    if env.get("LLM_MODE") == "local":
        slug = env.get("LLM_NAME_SLUG", "")
        sizing = repo / f"deploy/docker/services/nim/{slug}/hw-{hw}.env"
        if not sizing.is_file():
            available = sorted(
                p.name for p in (repo / f"deploy/docker/services/nim/{slug}").glob("hw-*.env")
            ) if slug else []
            errors.append(
                f"LLM_MODE=local needs {sizing.relative_to(repo)}, which does not exist "
                f"(compose fails with a bare 'no such file'). Available: {available or 'none'}"
            )
        if mode != "2d" or bp != "bp_wh":
            errors.append(
                "LLM_MODE=local is only valid with MODE=2d + BP_PROFILE=bp_wh; "
                f"got MODE={mode!r} + BP_PROFILE={bp!r}"
            )

    # 8. Warehouse uses the integrated RTVI VLM, never the standalone VLM NIM,
    # and the Docker path does not support pointing that VLM at a remote
    # endpoint either. `VLM_MODE=remote` is *exposed* -- blueprint-deploy.sh
    # takes --use-remote-vlm, sets VLM_BASE_URL/RTVI_VLM_ENDPOINT and
    # RTVI_VLM_MODEL_PATH=none -- but it never switches the two selectors that
    # decide which backend actually serves the request:
    #
    #   RTVI_VLM_MODEL_TO_USE  stays 'cosmos-reason3' (remote needs 'openai-compat')
    #   VLM_MODEL_TYPE         stays 'rtvi', so the agent keeps routing through
    #                          the misconfigured local RT-VLM proxy
    #
    # Both are set in industry-profiles/warehouse-operations/overrides.env and
    # neither is touched by --use-remote-vlm. The equivalent backend-selection
    # bug was fixed for Helm only (NVIDIA-AI-Blueprints/video-search-and-summarization#1501);
    # the Docker warehouse path was not updated, and no warehouse end-to-end run
    # has validated remote VLM. Rejecting it here fails fast on a knob that
    # looks supported and is not. Revisit when the Docker path sets both
    # selectors -- see references/profiles/warehouse.md.
    for key in ("VLM_MODE", "VLM_NAME_SLUG"):
        if env.get(key, "none") != "none":
            errors.append(
                f"{key}={env.get(key)!r} must be 'none' on warehouse; "
                "the blueprint uses the integrated RTVI VLM, not the standalone "
                "VLM NIM, and the Docker warehouse path does not wire a remote "
                "VLM end to end (RTVI_VLM_MODEL_TO_USE stays 'cosmos-reason3' "
                "instead of 'openai-compat' and VLM_MODEL_TYPE stays 'rtvi', so "
                "requests still route through the local RT-VLM proxy)"
            )

    # 9. Variant provenance, and the infra floor the prune must not remove.
    if not variant:
        errors.append("FOUNDATION_VARIANT is required when FOUNDATION=warehouse")
    else:
        baseline = parse_env(foundation_dir / "overrides.env").get(variant)
        if baseline is None:
            errors.append(
                f"FOUNDATION_VARIANT={variant!r} is not defined in "
                f"{(foundation_dir / 'overrides.env').name}"
            )

    # 9b. Mode coherence: a key whose mode token contradicts MODE means the list
    # was edited or the wrong variant expanded. This is the one
    # silent-wrong-data-plane failure the other checks cannot see.
    if profiles and mode in MODES:
        mismatched = {}
        for key in profiles:
            tokens = set(MODE_TOKEN.findall(key))
            if tokens and mode not in tokens:
                mismatched[key] = "/".join(sorted(tokens))
        if mismatched:
            detail = ", ".join(f"{k} (mode {v})" for k, v in sorted(mismatched.items()))
            errors.append(
                f"COMPOSE_PROFILES contains service keys whose mode contradicts "
                f"MODE={mode}: {detail}. These resolve and boot healthy while "
                "publishing to the wrong topic, so no later gate catches them"
            )
    if not profiles:
        # Guarding the floor check on a non-empty list would let the worst case
        # through silently: with COMPOSE_PROFILES empty or unset, Compose selects
        # no services, `up -d` starts nothing and still exits 0, and a gate that
        # only inspects the list's *contents* reports clean.
        errors.append(
            "COMPOSE_PROFILES is empty or unset. Compose would select no services, "
            "so `up -d` starts nothing and still exits 0. Expand the "
            "FOUNDATION_VARIANT list into COMPOSE_PROFILES as a literal."
        )
    else:
        required_floor = list(INFRA_FLOOR)
        if mode != "auto-calibration":
            required_floor += SDRC_FLOOR
        missing = [s for s in required_floor if s not in profiles]
        if missing:
            errors.append(
                "COMPOSE_PROFILES is missing warehouse infrastructure services that no "
                f"capability names and nothing boots without: {missing}. A build missing "
                "these resolves and validates cleanly, then fails at bring-up."
            )

    # 10. Calibration is checked in per sample dataset, NOT under VSS_DATA_DIR.
    # Compose bind-mounts it by path from the repo:
    #   warehouse-<mode>-app/calibration/sample-data/<SAMPLE_VIDEO_DATASET>/calibration.json
    # The shipped sample datasets already carry it, so nothing needs generating.
    # Only a custom dataset lacks it -- and a missing bind source makes Docker
    # create a directory where a file is expected.
    # MODE=auto-calibration is exempt: it *produces* a calibration for the chosen
    # dataset rather than consuming a shipped one, and there is no
    # warehouse-auto-calibration-app tree to look in.
    if mode in MODES and mode != "auto-calibration" and dataset:
        calib = (
            repo
            / "deploy/docker/industry-profiles/warehouse-operations"
            / f"warehouse-{mode}-app/calibration/sample-data"
            / dataset
            / "calibration.json"
        )
        if not calib.is_file():
            errors.append(
                f"no calibration for SAMPLE_VIDEO_DATASET={dataset!r} in {mode} mode: "
                f"{calib.relative_to(repo)} does not exist. Compose bind-mounts this "
                "path, so Docker will create a directory where a file is expected and "
                "perception will emit nothing. Generate it with "
                "vss-generate-video-calibration, or use a shipped sample dataset"
            )

    if mode == "3d" and variant.endswith("_MINIMAL"):
        warnings.append(
            "MODE=3d on a _MINIMAL list deploys no Elasticsearch, so the "
            "mdx-bev index is never persisted and BEV output cannot be verified"
        )

    for warning in warnings:
        print(f"WARNING: {warning}", file=sys.stderr)
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("build_dir", type=Path, help="path to _builds/<name>")
    parser.add_argument("--repo-root", required=True, type=Path)
    args = parser.parse_args()

    build_dir = args.build_dir
    repo = args.repo_root
    override = build_dir / "override.env"
    if not override.is_file():
        print(f"ERROR: {override} not found", file=sys.stderr)
        raise SystemExit(1)

    foundation = parse_env(override).get("FOUNDATION", "")
    if foundation != "warehouse":
        print(f"FOUNDATION={foundation!r} is not warehouse; nothing to check.")
        return

    foundation_dir = repo / "deploy/docker/industry-profiles/warehouse-operations"
    env = layered_env(repo, foundation_dir, build_dir)
    errors = check(env, repo, foundation_dir)

    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
    print(f"warehouse constraints OK: MODE={env.get('MODE')} "
          f"BP_PROFILE={env.get('BP_PROFILE')} variant={env.get('FOUNDATION_VARIANT')}")


if __name__ == "__main__":
    main()
