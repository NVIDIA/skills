---
name: error-handling
description: Error handling guidelines for NeMo-RL. Covers exception specificity, minimal try bodies, and else blocks.
when_to_use: Writing or reviewing exception handling; 'try-except', 'catch all exceptions', 'bare except', 'how to handle errors', during code review.
---

# Error Handling

## Use Specific Exceptions

When using try-except blocks, limit the except to the smallest set of errors possible.

**Don't:**
```python
try:
    open(path, "r").read()
except:
    print("Failed to open file")
```

**Do:**
```python
try:
    open(path, "r").read()
except FileNotFoundError:
    print("Failed to open file")
```
