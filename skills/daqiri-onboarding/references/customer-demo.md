# Optional Customer Demo

Only on an explicit demo request, use the prompt supplied by the user. Otherwise, check `demos/onboarding/prompt.txt` and expand the legacy glob `customer-demo-*/onboarding-demo-prompt.txt` from the repository root, not the skill installation directory. Inspect only these exact prompt paths; do not recursively search demo logs, assets or the checkout. If multiple candidates exist, ask which to use. Demo packages are optional and are not shipped with this skill. If none exists, explain that the customer demo is unavailable and offer ordinary onboarding; do not invent its sequence or start traffic.

When a prompt exists, read it and the documents it requests before beginning. Follow its latest presentation, authorization, preparation and readiness instructions over older demo notes. Saving, editing or explaining a prompt does not launch the demo. A demo request does not itself authorize privileged actions beyond the session's explicit scope.
