# NV-MONAI-Autoresearch Web UI

Read-only localhost file browser for a challenge folder.

```text
python3 "${SKILL_ROOT}/scripts/webui/app.py" --challenge challenge/<name> --port 8765
```

The left pane shows the challenge's file structure. Select a file to open it in
the right pane:

```text
*.md                    rendered Markdown
*.log, *.json, code     readable text
images and PDFs         native browser preview
*.html                  rendered page
```

Experiment folders expose their complete local record without another UI
abstraction:

```text
baselines/<baseline>/runs/<exp>/
  experiment.md  Intent before launch; Outcome after a valid result
  jobs/
  logs and result files
  result_viewer.html
```

Viewer HTML is served from a path-preserving `/files/...` URL so relative image
paths inside `result_viewer.html` resolve against the run folder.
The UI does not parse, summarize, rank, or write challenge state.
