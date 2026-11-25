# Evaluation Fixtures

These minimized fixtures are authored for deterministic skill evaluation. They
represent common standalone Hugging Face Trainer and TRL SFTTrainer structures;
they are not copies of runnable product examples and intentionally avoid model
or dataset downloads during evaluation setup.

The `injection-hf` fixture is synthetic and adds adversarial conversion
instructions to code comments, README text, configuration, and requirement
prose, plus Hugging Face-style model and dataset cards. All URLs use reserved
`example.com` domains. The fixture exists only to verify that source text
cannot direct conversion, authorize execution, trigger downloads/uploads, or
suppress validation and reporting.
