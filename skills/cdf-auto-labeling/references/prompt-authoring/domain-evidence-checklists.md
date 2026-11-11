# Domain Evidence Checklists

Use these checklists to make prompts domain-specific without encouraging
unsupported inference.

## Traffic / Roadway Safety

Focus on road layout, camera angle, lanes/shoulders/medians, vehicles,
pedestrians, cyclists, traffic lights only when visibly present, stopped or
disabled vehicles, debris, emergency/tow response, weather, lighting, and road
surface. Never invent signals, stop lines, lane numbers, collision victims, or
causality. Static incident aftermath should be distinguished from an observed
active collision.

## Warehouse / Operational Liability

Focus on workers, powered equipment, loads, aisles, loading docks, production
lines, PPE, blocked egress, wet floors, unstable pallets, spills, falling
objects, equipment/person intermixing, and visible near-misses. Do not infer
injury severity, fault, employment status, or legal liability.

## Security Surveillance

Focus on setting, entry/exit, visible access-control interaction, loitering,
object handling, concealment cues, confrontation, weapons only when visible, and
security/staff/law-enforcement response. Do not assert criminal intent or legal
categories; phrase uncertain behavior as visually consistent with a cue.

## Employee Conduct

Focus on work setting, role context visible from uniform/PPE/station, on-task
activity, off-task behavior, customer/colleague interactions, badge/PPE/dress
signals, and unattended station/equipment handling. Do not infer motivation,
employment status, discipline, gender, age, ethnicity, or skin tone.

## Robotics

Focus on robot morphology, end effector/tool, manipulated object, task phase,
contact state, success/failure evidence, safety envelope, human proximity, and
environment constraints. Do not invent robot intent or task success when the
result is occluded.

## Generic Images

Focus on objects, attributes, relationships, scene type, activity, image quality,
readable text, and uncertainty. Never describe motion or temporal outcomes for a
single still image.

## Person Attribute Images

Use only visible clothing, accessories, carried items, pose, crop quality, and
occlusion. Avoid identity, age, gender, ethnicity, skin tone, attractiveness,
health, or socioeconomic inference. Include uncertainty for occluded or cropped
regions.
