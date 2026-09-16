# Released KERMT models

Each released KERMT checkpoint is distributed as a **directory bundle**
containing the ckpt itself plus its vocab files:

```
<released_model>/
├── last_checkpoint.pt
├── pretrain_atom_vocab.{json,pkl}    # either extension; pkl in current releases
├── pretrain_bond_vocab.{json,pkl}    # either extension; pkl in current releases
└── pretrain_smiles_vocab.pkl         # only for cmim / hybrid ckpts (pickle-only)
```

If you're upgrading a grover_base ckpt to hybrid with
`kermt-add-cmim-pretrain`, the
upgrade step builds a fresh `pretrain_smiles_vocab.pkl` from your
pretrain corpus — released bundles only ship the smiles vocab for
already-cmim / already-hybrid ckpts.

The vocab files are an inseparable part of the released model — the ckpt's
vocab head dimensions are fixed at training time and only match these specific
vocab files. `kermt-continue-pretrain` treats the released ckpt's vocab as
authoritative: new corpora are tokenized through it rather than producing a
new vocab that would mismatch the ckpt's heads.

The skill auto-detects the three vocab files in the ckpt's parent directory
and passes them through `prepare_data.py --vocab-dir`. If the bundle is
incomplete (or the user has the ckpt alone), the skill asks for the
`--vocab-dir` path; if the user can't provide one, the skill refuses to
proceed and suggests `kermt-pretrain-scratch` instead.

To train a model on a corpus the released vocab can't cover, use
`kermt-pretrain-scratch` — the new vocab is built from the corpus and the
model is initialized fresh (no warm start; days-scale to converge).
