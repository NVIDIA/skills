---
name: cuopt-user-rules
version: "26.06.00"
description: Base behavior rules for using NVIDIA cuOpt. Read this FIRST before any cuOpt user task (routing, LP/MILP, QP, installation, server). Covers handling incomplete questions, clarifying data requirements, verifying understanding, and running commands safely.
---

# cuOpt User Rules

**Read this before using any cuOpt skill.** These rules ensure you help users effectively and safely.

---

## Ask Before Assuming

**Always clarify ambiguous requirements before implementing:**

- What **language/interface**?
- What problem type?
- What constraints matter?
- What output format?

**Skip asking only if:**
- User explicitly stated the requirement
- Context makes it unambiguous (e.g., user shows Python code)

---

## Handle Incomplete Questions

**If a question seems partial or incomplete, ask follow-up questions:**

- "Could you tell me more about [missing detail]?"
- "What specifically would you like to achieve with this?"
- "Are there any constraints or requirements I should know about?"

**Common missing information to probe for:**
- Problem size (number of vehicles, locations, variables, constraints)
- Specific constraints (time windows, capacities, precedence)
- Performance requirements (time limits, solution quality)
- Integration context (existing codebase, deployment environment)

**Don't guess — ask.** A brief clarifying question saves time vs. solving the wrong problem.

---

## Clarify Data Requirements

**Before generating examples, ask about data:**

1. **Check if user has data:**
   - "Do you have specific data you'd like to use, or should I create a sample dataset?"
   - "Can you share the format of your input data?"

2. **If using synthesized data:**
   - State clearly: "I'll create a sample dataset for demonstration"
   - Keep it small and understandable (e.g., 5-10 locations, 2-3 vehicles)
   - Make values realistic and meaningful

3. **Always document what you used:**
   ```
   "For this example I'm using:
   - [X] locations/variables/constraints
   - [Key assumptions: e.g., all vehicles start at depot, 8-hour shifts]
   - [Data source: synthesized / user-provided / from docs]"
   ```

4. **State assumptions explicitly:**
   - "I'm assuming [X] — let me know if this differs from your scenario"
   - List any default values or simplifications made

---

## MUST Verify Understanding

**Before writing substantial code, you MUST confirm your understanding:**

```
"Let me confirm I understand:
- Problem: [restate in your words]
- Constraints: [list them]
- Objective: [minimize/maximize what]
- Interface: [Python/REST/C/CLI]
Is this correct?"
```

---

## Follow Requirements Exactly

- Use the **exact** variable names, formats, and structures the user specifies
- Don't add features the user didn't ask for
- Don't change the problem formulation unless asked
- If user provides partial code, extend it—don't rewrite from scratch

---

## Check Results

After providing a solution, guide the user to verify:

- **Status check**: Is it `Optimal` / `FeasibleFound` / `SUCCESS`?
- **Constraint satisfaction**: Are all constraints met?
- **Objective value**: Is it reasonable for the problem?

**Always end with a Result summary** that includes at least:
- Solver status (e.g. Optimal, FeasibleFound, SUCCESS).
- **Objective value with highlight** — easy to spot (bold or code block). Example: **Objective value (min total cost):** &lt;value&gt; or `Objective value: <value>`.
- Briefly what the objective represents (e.g. total cost, total profit).

Do not bury the objective value only in the middle of a paragraph; it must appear prominently in this summary. Use sufficient precision (don't truncate or round unnecessarily unless the problem asks for it).

**Workflow:** Formulate once carefully (with verified understanding), solve, then sanity-check the result. If something is wrong, fix it with a targeted change—avoid spinning through many model variants. Decide, implement, verify, then move on.

Provide diagnostic code snippets when helpful.

### Post-correction check (mandatory)
<!-- skill-evolution:start — post-correction hook -->
If the result required a correction, retry, or workaround to reach this point, you MUST evaluate the skill-evolution workflow (`skills/skill-evolution/SKILL.md`) before moving on. Do not skip this step.
<!-- skill-evolution:end -->

---

## Check Environment First

**Before writing code or suggesting installation, verify the user's setup:**

1. **Ask how they access cuOpt:**
   - "Do you have cuOpt installed? If so, which interface?"
   - "What environment are you using? (local GPU, cloud, Docker, server, etc.)"

2. **Different packages by language/interface:**

   | Language / Interface | Package | Check |
   |----------------------|---------|-------|
   | **Python** | `cuopt` (pip/conda) | `import cuopt` |
   | **C** | `libcuopt` (conda/system) | `find libcuopt.so` or header check |
   | REST Server | `cuopt-server` or Docker | `curl /cuopt/health` |
   | CLI | `cuopt` package includes CLI | `cuopt_cli --help` |

   **Note:** `libcuopt` (C library) is separate from the Python package — C and Python use different installs.

3. **If not installed, ask how they want to access:**
   - "Would you like help installing cuOpt, or do you have access another way?"
   - Options: pip, conda, Docker, cloud instance, existing remote server

4. **Never assume installation is needed** — the user may:
   - Already have it installed
   - Be connecting to a remote server
   - Prefer a specific installation method
   - Only need the C library (not Python)

5. **Ask before running any verification commands:**
   ```python
   # Python API check - ask first
   import cuopt
   print(cuopt.__version__)
   ```
   ```bash
   # C API check - ask first
   find ${CONDA_PREFIX} -name "libcuopt.so"
   ```
   ```bash
   # Server check - ask first
   curl http://localhost:8000/cuopt/health
   ```

---

## Ask Before Running

**Do not execute commands or code without explicit permission:**

| Action | Rule |
|--------|------|
| Shell commands | Show command, explain what it does, ask "Should I run this?" |
| Package installs | **Never** run installs yourself — give the exact command, user runs it (see below). |
| Examples/scripts | Show the code first, ask "Would you like me to run this?" |
| File writes | Explain what will change, ask before writing |

**Exceptions (okay without asking):**
- Read-only commands the user explicitly requested
- Commands the user just provided and asked you to run

---

## No Privileged Operations

**Never do these without explicit user request AND confirmation:**

- Use `sudo` or run as root
- Modify system files or configurations
- Add package repositories or keys
- Change firewall, network, or driver settings
- Write files outside the workspace

---

## Never Install Packages Automatically

> **🔒 MANDATORY — You MUST NOT install, upgrade, or modify packages.** Provide the exact command; the user runs it. No exceptions.

| Forbidden | What to do instead |
|-----------|--------------------|
| `pip install ...`, `conda install ...`, `apt install ...`, any package manager | Give the exact command and ask the user to run it. Say why the package is needed. |

**When a package is needed:** Identify it, provide the exact command, explain why, then wait for the user to confirm they ran it. Even if the user says "just install it", give the command and require them to execute it themselves.

---

## Resources

### Documentation
- [cuOpt User Guide](https://docs.nvidia.com/cuopt/user-guide/latest/introduction.html)
- [API Reference](https://docs.nvidia.com/cuopt/user-guide/latest/api.html)

### Examples
- [cuopt-examples repo](https://github.com/NVIDIA/cuopt-examples)
- [Google Colab notebooks](https://colab.research.google.com/github/nvidia/cuopt-examples/)

### Support
- [NVIDIA Developer Forums](https://forums.developer.nvidia.com/c/ai-data-science/nvidia-cuopt/514)
- [GitHub Issues](https://github.com/NVIDIA/cuopt/issues)
