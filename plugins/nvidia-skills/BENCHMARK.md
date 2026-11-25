# Plugin benchmark

> **Overall verdict: PASS**

## Publication Recommendation

- Evaluation source: self-assessed

CI gate decision: allow. Supplied CI context; this renderer does not authenticate the gate.

Canonical assessment: PASS

Required validation: PASS. Evaluation state: supplied.

- Agents: `claude`, `skill_agent`

**Partial** — current quality uses available evidence; coverage is incomplete.

Current quality: PASS; score 82.71/100; scored agents 2/2.

Effectiveness has a failing score\. · Issues found in Correctness and Efficiency graders · Not all grader results are available

Canonical lift: PASS; +23.21 points. The best agent&#x27;s average uplift is at least 5 points\.

- Effectiveness has a failing score\.

- Issues found in Correctness

- Efficiency graders

- Not all grader results are available

## Results at a Glance

Cells with arrows show baseline → current and lift over the producer’s matched population. Other cells show current quality. Full current-quality coverage is shown separately below.

| Dimension | `claude` | `skill_agent` |
| --- | --- | --- |
| Security | 100.00% → 100.00% (+0.00 points) | 100.00% → 100.00% (+0.00 points) |
| Correctness | 34.26% → 87.96% (+53.70 points) | 58.33% → 94.44% (+36.11 points) |
| Discoverability | 100.00% | 100.00% |
| Effectiveness | 33.33% | 52.78% |
| Efficiency | 76.29% → 92.23% (+15.94 points) | 100.00% → 100.00% (+0.00 points) |

### Current quality and coverage

| Agent | Score /100 | Verdict | Coverage | Scored dimensions |
| --- | --- | --- | --- | --- |
| claude | 82\.71 | FAIL | partial | 5/5 |
| skill\_agent | 89\.44 | PASS | partial | 5/5 |

| Agent | Dimension | Current score | Verdict | Coverage | Observations |
| --- | --- | --- | --- | --- | --- |
| claude | security | 100\.00% | PASS | partial | inapplicable\: 0, ineligible\: 0, measured\: 5, unresolved\: 13 |
| claude | correctness | 87\.96% | PASS | complete | inapplicable\: 0, ineligible\: 0, measured\: 18, unresolved\: 0 |
| claude | discoverability | 100\.00% | PASS | complete | inapplicable\: 0, ineligible\: 0, measured\: 18, unresolved\: 0 |
| claude | effectiveness | 33\.33% | FAIL | partial | inapplicable\: 5, ineligible\: 0, measured\: 10, unresolved\: 3 |
| claude | efficiency | 92\.23% | PASS | complete | inapplicable\: 0, ineligible\: 0, measured\: 18, unresolved\: 0 |
| skill\_agent | security | 100\.00% | PASS | partial | inapplicable\: 0, ineligible\: 0, measured\: 5, unresolved\: 13 |
| skill\_agent | correctness | 94\.44% | PASS | complete | inapplicable\: 0, ineligible\: 0, measured\: 18, unresolved\: 0 |
| skill\_agent | discoverability | 100\.00% | PASS | complete | inapplicable\: 0, ineligible\: 0, measured\: 18, unresolved\: 0 |
| skill\_agent | effectiveness | 52\.78% | PASS | partial | inapplicable\: 5, ineligible\: 0, measured\: 12, unresolved\: 1 |
| skill\_agent | efficiency | 100\.00% | PASS | complete | inapplicable\: 0, ineligible\: 0, measured\: 18, unresolved\: 0 |

### Matched baseline coverage

| Agent | Dimension | Included / expected pairs | State |
| --- | --- | --- | --- |
| claude | security | 4/18 | measured |
| claude | correctness | 18/18 | measured |
| claude | effectiveness | 0/18 | unavailable |
| claude | efficiency | 18/18 | measured |
| skill\_agent | security | 5/18 | measured |
| skill\_agent | correctness | 18/18 | measured |
| skill\_agent | effectiveness | 0/18 | unavailable |
| skill\_agent | efficiency | 18/18 | measured |

<details>
<summary>Recorded token usage and execution cost</summary>

These are original per-attempt measurements, including recorded variants and retries. Token usage is separate from the Efficiency score; it does not measure added context cost.

| Result | Agent | Variant | Measurement | Value | Unit | Status |
| --- | --- | --- | --- | --- | --- | --- |
| 580a70ce98b74e1d8791dbb792e9227e | codex\_agent | skill | input\_tokens | 68983\.0 | tokens | No errors reported |
| 580a70ce98b74e1d8791dbb792e9227e | codex\_agent | skill | output\_tokens | 649\.0 | tokens | No errors reported |
| 580a70ce98b74e1d8791dbb792e9227e | codex\_agent | skill | cached\_tokens | 53591\.0 | tokens | No errors reported |
| 580a70ce98b74e1d8791dbb792e9227e | codex\_agent | skill | reasoning\_tokens | 0\.0 | tokens | No errors reported |
| 4ccbc07b69fa4f42ae3b5ad016d5d9c6 | codex\_agent | baseline | input\_tokens | 68651\.0 | tokens | No errors reported |
| 4ccbc07b69fa4f42ae3b5ad016d5d9c6 | codex\_agent | baseline | output\_tokens | 572\.0 | tokens | No errors reported |
| 4ccbc07b69fa4f42ae3b5ad016d5d9c6 | codex\_agent | baseline | cached\_tokens | 52697\.0 | tokens | No errors reported |
| 4ccbc07b69fa4f42ae3b5ad016d5d9c6 | codex\_agent | baseline | reasoning\_tokens | 13\.0 | tokens | No errors reported |
| 62d5321741764c8ba4003d0960ec893a | claude\_code\_agent | skill | input\_tokens | 126854\.0 | tokens | No errors reported |
| 62d5321741764c8ba4003d0960ec893a | claude\_code\_agent | skill | output\_tokens | 1195\.0 | tokens | No errors reported |
| 62d5321741764c8ba4003d0960ec893a | claude\_code\_agent | skill | cached\_tokens | 106160\.0 | tokens | No errors reported |
| a85228a3dfe7486dbe7b40a16505741c | claude\_code\_agent | baseline | input\_tokens | 132519\.0 | tokens | No errors reported |
| a85228a3dfe7486dbe7b40a16505741c | claude\_code\_agent | baseline | output\_tokens | 1577\.0 | tokens | No errors reported |
| a85228a3dfe7486dbe7b40a16505741c | claude\_code\_agent | baseline | cached\_tokens | 121171\.0 | tokens | No errors reported |
| 21294700add1464b9b7e80df780c5894 | codex\_agent | skill | input\_tokens | 70597\.0 | tokens | No errors reported |
| 21294700add1464b9b7e80df780c5894 | codex\_agent | skill | output\_tokens | 784\.0 | tokens | No errors reported |
| 21294700add1464b9b7e80df780c5894 | codex\_agent | skill | cached\_tokens | 53692\.0 | tokens | No errors reported |
| 21294700add1464b9b7e80df780c5894 | codex\_agent | skill | reasoning\_tokens | 56\.0 | tokens | No errors reported |
| 3425ae98ede94c70a0d05e15f939154d | codex\_agent | baseline | input\_tokens | 11517\.0 | tokens | No errors reported |
| 3425ae98ede94c70a0d05e15f939154d | codex\_agent | baseline | output\_tokens | 440\.0 | tokens | No errors reported |
| 3425ae98ede94c70a0d05e15f939154d | codex\_agent | baseline | cached\_tokens | 5665\.0 | tokens | No errors reported |
| 3425ae98ede94c70a0d05e15f939154d | codex\_agent | baseline | reasoning\_tokens | 82\.0 | tokens | No errors reported |
| 97f585060812437ab6b797f67bda51f5 | claude\_code\_agent | skill | input\_tokens | 126866\.0 | tokens | No errors reported |
| 97f585060812437ab6b797f67bda51f5 | claude\_code\_agent | skill | output\_tokens | 1047\.0 | tokens | No errors reported |
| 97f585060812437ab6b797f67bda51f5 | claude\_code\_agent | skill | cached\_tokens | 106268\.0 | tokens | No errors reported |
| 6421d383ffac4fbf811e242d19b28160 | claude\_code\_agent | baseline | input\_tokens | 64730\.0 | tokens | No errors reported |
| 6421d383ffac4fbf811e242d19b28160 | claude\_code\_agent | baseline | output\_tokens | 1081\.0 | tokens | No errors reported |
| 6421d383ffac4fbf811e242d19b28160 | claude\_code\_agent | baseline | cached\_tokens | 54458\.0 | tokens | No errors reported |
| fb8cfe71ac6e48b9a9ec002a0414b753 | codex\_agent | skill | input\_tokens | 70025\.0 | tokens | No errors reported |
| fb8cfe71ac6e48b9a9ec002a0414b753 | codex\_agent | skill | output\_tokens | 655\.0 | tokens | No errors reported |
| fb8cfe71ac6e48b9a9ec002a0414b753 | codex\_agent | skill | cached\_tokens | 53541\.0 | tokens | No errors reported |
| fb8cfe71ac6e48b9a9ec002a0414b753 | codex\_agent | skill | reasoning\_tokens | 0\.0 | tokens | No errors reported |
| 3bbd1837aa094b22bdd77a34d9073b8b | codex\_agent | baseline | input\_tokens | 28175\.0 | tokens | No errors reported |
| 3bbd1837aa094b22bdd77a34d9073b8b | codex\_agent | baseline | output\_tokens | 521\.0 | tokens | No errors reported |
| 3bbd1837aa094b22bdd77a34d9073b8b | codex\_agent | baseline | cached\_tokens | 19895\.0 | tokens | No errors reported |
| 3bbd1837aa094b22bdd77a34d9073b8b | codex\_agent | baseline | reasoning\_tokens | 25\.0 | tokens | No errors reported |
| d5ae2d2886cd420c80e3d07faae6b5ec | claude\_code\_agent | skill | input\_tokens | 127554\.0 | tokens | No errors reported |
| d5ae2d2886cd420c80e3d07faae6b5ec | claude\_code\_agent | skill | output\_tokens | 1287\.0 | tokens | No errors reported |
| d5ae2d2886cd420c80e3d07faae6b5ec | claude\_code\_agent | skill | cached\_tokens | 106637\.0 | tokens | No errors reported |
| 799e3b9c29184681bb4463ec70b33e39 | claude\_code\_agent | baseline | input\_tokens | 42956\.0 | tokens | No errors reported |
| 799e3b9c29184681bb4463ec70b33e39 | claude\_code\_agent | baseline | output\_tokens | 623\.0 | tokens | No errors reported |
| 799e3b9c29184681bb4463ec70b33e39 | claude\_code\_agent | baseline | cached\_tokens | 33017\.0 | tokens | No errors reported |
| b727efd66fcd41b28fe281ba61b002c3 | codex\_agent | skill | input\_tokens | 69724\.0 | tokens | No errors reported |
| b727efd66fcd41b28fe281ba61b002c3 | codex\_agent | skill | output\_tokens | 657\.0 | tokens | No errors reported |
| b727efd66fcd41b28fe281ba61b002c3 | codex\_agent | skill | cached\_tokens | 53869\.0 | tokens | No errors reported |
| b727efd66fcd41b28fe281ba61b002c3 | codex\_agent | skill | reasoning\_tokens | 21\.0 | tokens | No errors reported |
| a9452a89c3e642b09cca04cedfba2d86 | codex\_agent | baseline | input\_tokens | 17596\.0 | tokens | No errors reported |
| a9452a89c3e642b09cca04cedfba2d86 | codex\_agent | baseline | output\_tokens | 414\.0 | tokens | No errors reported |
| a9452a89c3e642b09cca04cedfba2d86 | codex\_agent | baseline | cached\_tokens | 11540\.0 | tokens | No errors reported |
| a9452a89c3e642b09cca04cedfba2d86 | codex\_agent | baseline | reasoning\_tokens | 50\.0 | tokens | No errors reported |
| 7343541952bc438f937ca2d43bbf4ee3 | claude\_code\_agent | skill | input\_tokens | 284620\.0 | tokens | No errors reported |
| 7343541952bc438f937ca2d43bbf4ee3 | claude\_code\_agent | skill | output\_tokens | 2773\.0 | tokens | No errors reported |
| 7343541952bc438f937ca2d43bbf4ee3 | claude\_code\_agent | skill | cached\_tokens | 211244\.0 | tokens | No errors reported |
| 467619b659554722b71fdbd7c27bff95 | claude\_code\_agent | baseline | input\_tokens | 64926\.0 | tokens | No errors reported |
| 467619b659554722b71fdbd7c27bff95 | claude\_code\_agent | baseline | output\_tokens | 1614\.0 | tokens | No errors reported |
| 467619b659554722b71fdbd7c27bff95 | claude\_code\_agent | baseline | cached\_tokens | 54729\.0 | tokens | No errors reported |
| 5d45fc621a8c41fbb02c7b2dcc67c600 | codex\_agent | skill | input\_tokens | 74083\.0 | tokens | No errors reported |
| 5d45fc621a8c41fbb02c7b2dcc67c600 | codex\_agent | skill | output\_tokens | 937\.0 | tokens | No errors reported |
| 5d45fc621a8c41fbb02c7b2dcc67c600 | codex\_agent | skill | cached\_tokens | 53557\.0 | tokens | No errors reported |
| 5d45fc621a8c41fbb02c7b2dcc67c600 | codex\_agent | skill | reasoning\_tokens | 100\.0 | tokens | No errors reported |
| 464d5182f1a64a47a23450d48920b2fa | codex\_agent | baseline | input\_tokens | 93816\.0 | tokens | No errors reported |
| 464d5182f1a64a47a23450d48920b2fa | codex\_agent | baseline | output\_tokens | 1129\.0 | tokens | No errors reported |
| 464d5182f1a64a47a23450d48920b2fa | codex\_agent | baseline | cached\_tokens | 71647\.0 | tokens | No errors reported |
| 464d5182f1a64a47a23450d48920b2fa | codex\_agent | baseline | reasoning\_tokens | 17\.0 | tokens | No errors reported |
| aafa4d0b697a4468b47a1383a170b531 | claude\_code\_agent | skill | input\_tokens | 127098\.0 | tokens | No errors reported |
| aafa4d0b697a4468b47a1383a170b531 | claude\_code\_agent | skill | output\_tokens | 1425\.0 | tokens | No errors reported |
| aafa4d0b697a4468b47a1383a170b531 | claude\_code\_agent | skill | cached\_tokens | 106138\.0 | tokens | No errors reported |
| 2e2d116a4b2b4d0ebb07d8d229dead2b | claude\_code\_agent | baseline | input\_tokens | 65032\.0 | tokens | No errors reported |
| 2e2d116a4b2b4d0ebb07d8d229dead2b | claude\_code\_agent | baseline | output\_tokens | 876\.0 | tokens | No errors reported |
| 2e2d116a4b2b4d0ebb07d8d229dead2b | claude\_code\_agent | baseline | cached\_tokens | 54745\.0 | tokens | No errors reported |
| b8dc89d8514f4919b7ec1569639ec956 | codex\_agent | skill | input\_tokens | 70868\.0 | tokens | No errors reported |
| b8dc89d8514f4919b7ec1569639ec956 | codex\_agent | skill | output\_tokens | 758\.0 | tokens | No errors reported |
| b8dc89d8514f4919b7ec1569639ec956 | codex\_agent | skill | cached\_tokens | 53911\.0 | tokens | No errors reported |
| b8dc89d8514f4919b7ec1569639ec956 | codex\_agent | skill | reasoning\_tokens | 31\.0 | tokens | No errors reported |
| 7c0f2043f5b642fc85e6d0f1a9306bae | codex\_agent | baseline | input\_tokens | 11582\.0 | tokens | No errors reported |
| 7c0f2043f5b642fc85e6d0f1a9306bae | codex\_agent | baseline | output\_tokens | 219\.0 | tokens | No errors reported |
| 7c0f2043f5b642fc85e6d0f1a9306bae | codex\_agent | baseline | cached\_tokens | 5672\.0 | tokens | No errors reported |
| 7c0f2043f5b642fc85e6d0f1a9306bae | codex\_agent | baseline | reasoning\_tokens | 0\.0 | tokens | No errors reported |
| b4def1ce76fc43bbbb41308c85e719aa | claude\_code\_agent | skill | input\_tokens | 127697\.0 | tokens | No errors reported |
| b4def1ce76fc43bbbb41308c85e719aa | claude\_code\_agent | skill | output\_tokens | 1136\.0 | tokens | No errors reported |
| b4def1ce76fc43bbbb41308c85e719aa | claude\_code\_agent | skill | cached\_tokens | 106791\.0 | tokens | No errors reported |
| f724716662254ec4ad3c8f0fcfbf65d4 | claude\_code\_agent | baseline | input\_tokens | 64972\.0 | tokens | No errors reported |
| f724716662254ec4ad3c8f0fcfbf65d4 | claude\_code\_agent | baseline | output\_tokens | 928\.0 | tokens | No errors reported |
| f724716662254ec4ad3c8f0fcfbf65d4 | claude\_code\_agent | baseline | cached\_tokens | 54703\.0 | tokens | No errors reported |
| d95e9fffdd824b09ade8ab2c8b86ac0b | codex\_agent | skill | input\_tokens | 53806\.0 | tokens | No errors reported |
| d95e9fffdd824b09ade8ab2c8b86ac0b | codex\_agent | skill | output\_tokens | 871\.0 | tokens | No errors reported |
| d95e9fffdd824b09ade8ab2c8b86ac0b | codex\_agent | skill | cached\_tokens | 39513\.0 | tokens | No errors reported |
| d95e9fffdd824b09ade8ab2c8b86ac0b | codex\_agent | skill | reasoning\_tokens | 84\.0 | tokens | No errors reported |
| da2935fbf8ed4d65ade44a963f162802 | codex\_agent | baseline | input\_tokens | 11681\.0 | tokens | No errors reported |
| da2935fbf8ed4d65ade44a963f162802 | codex\_agent | baseline | output\_tokens | 524\.0 | tokens | No errors reported |
| da2935fbf8ed4d65ade44a963f162802 | codex\_agent | baseline | cached\_tokens | 5672\.0 | tokens | No errors reported |
| da2935fbf8ed4d65ade44a963f162802 | codex\_agent | baseline | reasoning\_tokens | 115\.0 | tokens | No errors reported |
| 01a5e7e3fab9420c956e420a4ab8fd42 | claude\_code\_agent | skill | input\_tokens | 127388\.0 | tokens | No errors reported |
| 01a5e7e3fab9420c956e420a4ab8fd42 | claude\_code\_agent | skill | output\_tokens | 2125\.0 | tokens | No errors reported |
| 01a5e7e3fab9420c956e420a4ab8fd42 | claude\_code\_agent | skill | cached\_tokens | 106260\.0 | tokens | No errors reported |
| 2b9f98eca80c42b69a3376cfd65791b1 | claude\_code\_agent | baseline | input\_tokens | 214625\.0 | tokens | No errors reported |
| 2b9f98eca80c42b69a3376cfd65791b1 | claude\_code\_agent | baseline | output\_tokens | 11672\.0 | tokens | No errors reported |
| 2b9f98eca80c42b69a3376cfd65791b1 | claude\_code\_agent | baseline | cached\_tokens | 192951\.0 | tokens | No errors reported |
| 6913cabfe55f4161a1c5a9c708502025 | codex\_agent | skill | input\_tokens | 71410\.0 | tokens | No errors reported |
| 6913cabfe55f4161a1c5a9c708502025 | codex\_agent | skill | output\_tokens | 803\.0 | tokens | No errors reported |
| 6913cabfe55f4161a1c5a9c708502025 | codex\_agent | skill | cached\_tokens | 53344\.0 | tokens | No errors reported |
| 6913cabfe55f4161a1c5a9c708502025 | codex\_agent | skill | reasoning\_tokens | 24\.0 | tokens | No errors reported |
| ee4f3c9788be4b7f8c01359b86beb0d2 | codex\_agent | baseline | input\_tokens | 5676\.0 | tokens | No errors reported |
| ee4f3c9788be4b7f8c01359b86beb0d2 | codex\_agent | baseline | output\_tokens | 284\.0 | tokens | No errors reported |
| ee4f3c9788be4b7f8c01359b86beb0d2 | codex\_agent | baseline | cached\_tokens | 0\.0 | tokens | No errors reported |
| ee4f3c9788be4b7f8c01359b86beb0d2 | codex\_agent | baseline | reasoning\_tokens | 42\.0 | tokens | No errors reported |
| 865184f269c2454ca8907049db93e192 | claude\_code\_agent | skill | input\_tokens | 126960\.0 | tokens | No errors reported |
| 865184f269c2454ca8907049db93e192 | claude\_code\_agent | skill | output\_tokens | 1656\.0 | tokens | No errors reported |
| 865184f269c2454ca8907049db93e192 | claude\_code\_agent | skill | cached\_tokens | 106108\.0 | tokens | No errors reported |
| 6cb436ba493d4fc2b2818b7084b45267 | claude\_code\_agent | baseline | input\_tokens | 21246\.0 | tokens | No errors reported |
| 6cb436ba493d4fc2b2818b7084b45267 | claude\_code\_agent | baseline | output\_tokens | 518\.0 | tokens | No errors reported |
| 6cb436ba493d4fc2b2818b7084b45267 | claude\_code\_agent | baseline | cached\_tokens | 11755\.0 | tokens | No errors reported |
| 2ed0734842084bd584baa81ab4fa0cfb | codex\_agent | skill | input\_tokens | 53505\.0 | tokens | No errors reported |
| 2ed0734842084bd584baa81ab4fa0cfb | codex\_agent | skill | output\_tokens | 706\.0 | tokens | No errors reported |
| 2ed0734842084bd584baa81ab4fa0cfb | codex\_agent | skill | cached\_tokens | 39210\.0 | tokens | No errors reported |
| 2ed0734842084bd584baa81ab4fa0cfb | codex\_agent | skill | reasoning\_tokens | 16\.0 | tokens | No errors reported |
| 7bd6053bfb274cf7b805eacf90cf075e | codex\_agent | baseline | input\_tokens | 17492\.0 | tokens | No errors reported |
| 7bd6053bfb274cf7b805eacf90cf075e | codex\_agent | baseline | output\_tokens | 388\.0 | tokens | No errors reported |
| 7bd6053bfb274cf7b805eacf90cf075e | codex\_agent | baseline | cached\_tokens | 11508\.0 | tokens | No errors reported |
| 7bd6053bfb274cf7b805eacf90cf075e | codex\_agent | baseline | reasoning\_tokens | 0\.0 | tokens | No errors reported |
| 63b77204c5454e97a0b6bd012b4af696 | claude\_code\_agent | skill | input\_tokens | 127040\.0 | tokens | No errors reported |
| 63b77204c5454e97a0b6bd012b4af696 | claude\_code\_agent | skill | output\_tokens | 1433\.0 | tokens | No errors reported |
| 63b77204c5454e97a0b6bd012b4af696 | claude\_code\_agent | skill | cached\_tokens | 106094\.0 | tokens | No errors reported |
| a5f8105c9dbd40278009a1601e6ced35 | claude\_code\_agent | baseline | input\_tokens | 42782\.0 | tokens | No errors reported |
| a5f8105c9dbd40278009a1601e6ced35 | claude\_code\_agent | baseline | output\_tokens | 716\.0 | tokens | No errors reported |
| a5f8105c9dbd40278009a1601e6ced35 | claude\_code\_agent | baseline | cached\_tokens | 32998\.0 | tokens | No errors reported |
| 1c929a9ebf82475480451447315e0ddb | codex\_agent | skill | input\_tokens | 118247\.0 | tokens | No errors reported |
| 1c929a9ebf82475480451447315e0ddb | codex\_agent | skill | output\_tokens | 1863\.0 | tokens | No errors reported |
| 1c929a9ebf82475480451447315e0ddb | codex\_agent | skill | cached\_tokens | 93560\.0 | tokens | No errors reported |
| 1c929a9ebf82475480451447315e0ddb | codex\_agent | skill | reasoning\_tokens | 78\.0 | tokens | No errors reported |
| 896e75851d45422cb2b2b8ac6392de3c | codex\_agent | baseline | input\_tokens | 42117\.0 | tokens | No errors reported |
| 896e75851d45422cb2b2b8ac6392de3c | codex\_agent | baseline | output\_tokens | 1736\.0 | tokens | No errors reported |
| 896e75851d45422cb2b2b8ac6392de3c | codex\_agent | baseline | cached\_tokens | 32019\.0 | tokens | No errors reported |
| 896e75851d45422cb2b2b8ac6392de3c | codex\_agent | baseline | reasoning\_tokens | 58\.0 | tokens | No errors reported |
| cd9104cfafb54dba8ec5f2a3cc00ab67 | claude\_code\_agent | skill | input\_tokens | 159966\.0 | tokens | No errors reported |
| cd9104cfafb54dba8ec5f2a3cc00ab67 | claude\_code\_agent | skill | output\_tokens | 3194\.0 | tokens | No errors reported |
| cd9104cfafb54dba8ec5f2a3cc00ab67 | claude\_code\_agent | skill | cached\_tokens | 131694\.0 | tokens | No errors reported |
| df46a089b9094a22ad2225d935a058f5 | claude\_code\_agent | baseline | input\_tokens | 21238\.0 | tokens | No errors reported |
| df46a089b9094a22ad2225d935a058f5 | claude\_code\_agent | baseline | output\_tokens | 1520\.0 | tokens | No errors reported |
| df46a089b9094a22ad2225d935a058f5 | claude\_code\_agent | baseline | cached\_tokens | 11755\.0 | tokens | No errors reported |
| c56c65d0e0644c289ed5d874fc59cb70 | codex\_agent | skill | input\_tokens | 18164\.0 | tokens | No errors reported |
| c56c65d0e0644c289ed5d874fc59cb70 | codex\_agent | skill | output\_tokens | 481\.0 | tokens | No errors reported |
| c56c65d0e0644c289ed5d874fc59cb70 | codex\_agent | skill | cached\_tokens | 11945\.0 | tokens | No errors reported |
| c56c65d0e0644c289ed5d874fc59cb70 | codex\_agent | skill | reasoning\_tokens | 282\.0 | tokens | No errors reported |
| 9fdc9c6bd8694335893c48b810c43143 | codex\_agent | baseline | input\_tokens | 11571\.0 | tokens | No errors reported |
| 9fdc9c6bd8694335893c48b810c43143 | codex\_agent | baseline | output\_tokens | 240\.0 | tokens | No errors reported |
| 9fdc9c6bd8694335893c48b810c43143 | codex\_agent | baseline | cached\_tokens | 5662\.0 | tokens | No errors reported |
| 9fdc9c6bd8694335893c48b810c43143 | codex\_agent | baseline | reasoning\_tokens | 68\.0 | tokens | No errors reported |
| dab43e3ad6044928a5e52c463f713b10 | claude\_code\_agent | skill | input\_tokens | 109848\.0 | tokens | No errors reported |
| dab43e3ad6044928a5e52c463f713b10 | claude\_code\_agent | skill | output\_tokens | 1259\.0 | tokens | No errors reported |
| dab43e3ad6044928a5e52c463f713b10 | claude\_code\_agent | skill | cached\_tokens | 98666\.0 | tokens | No errors reported |
| d38432df48c64489b45ccdc0a1d6972e | claude\_code\_agent | baseline | input\_tokens | 175643\.0 | tokens | No errors reported |
| d38432df48c64489b45ccdc0a1d6972e | claude\_code\_agent | baseline | output\_tokens | 1470\.0 | tokens | No errors reported |
| d38432df48c64489b45ccdc0a1d6972e | claude\_code\_agent | baseline | cached\_tokens | 164528\.0 | tokens | No errors reported |
| c0b2a5cb6bf1458f83bc27be288ae7e7 | codex\_agent | skill | input\_tokens | 18222\.0 | tokens | No errors reported |
| c0b2a5cb6bf1458f83bc27be288ae7e7 | codex\_agent | skill | output\_tokens | 294\.0 | tokens | No errors reported |
| c0b2a5cb6bf1458f83bc27be288ae7e7 | codex\_agent | skill | cached\_tokens | 11971\.0 | tokens | No errors reported |
| c0b2a5cb6bf1458f83bc27be288ae7e7 | codex\_agent | skill | reasoning\_tokens | 0\.0 | tokens | No errors reported |
| 8d65a86490234879bc8441b3e821972c | codex\_agent | baseline | input\_tokens | 17669\.0 | tokens | No errors reported |
| 8d65a86490234879bc8441b3e821972c | codex\_agent | baseline | output\_tokens | 361\.0 | tokens | No errors reported |
| 8d65a86490234879bc8441b3e821972c | codex\_agent | baseline | cached\_tokens | 11562\.0 | tokens | No errors reported |
| 8d65a86490234879bc8441b3e821972c | codex\_agent | baseline | reasoning\_tokens | 19\.0 | tokens | No errors reported |
| 88b40d57d26c4476bdd09b2bda543ad8 | claude\_code\_agent | skill | input\_tokens | 87996\.0 | tokens | No errors reported |
| 88b40d57d26c4476bdd09b2bda543ad8 | claude\_code\_agent | skill | output\_tokens | 885\.0 | tokens | No errors reported |
| 88b40d57d26c4476bdd09b2bda543ad8 | claude\_code\_agent | skill | cached\_tokens | 76922\.0 | tokens | No errors reported |
| d08718b1506042cf82272cd07077ca41 | claude\_code\_agent | baseline | input\_tokens | 821649\.0 | tokens | No errors reported |
| d08718b1506042cf82272cd07077ca41 | claude\_code\_agent | baseline | output\_tokens | 13687\.0 | tokens | No errors reported |
| d08718b1506042cf82272cd07077ca41 | claude\_code\_agent | baseline | cached\_tokens | 790653\.0 | tokens | No errors reported |
| 7a8b15099a4e4d72a429472683c0fab9 | codex\_agent | skill | input\_tokens | 18349\.0 | tokens | No errors reported |
| 7a8b15099a4e4d72a429472683c0fab9 | codex\_agent | skill | output\_tokens | 370\.0 | tokens | No errors reported |
| 7a8b15099a4e4d72a429472683c0fab9 | codex\_agent | skill | cached\_tokens | 12025\.0 | tokens | No errors reported |
| 7a8b15099a4e4d72a429472683c0fab9 | codex\_agent | skill | reasoning\_tokens | 0\.0 | tokens | No errors reported |
| 163553242cbd42f88de528586ea41141 | codex\_agent | baseline | input\_tokens | 11639\.0 | tokens | No errors reported |
| 163553242cbd42f88de528586ea41141 | codex\_agent | baseline | output\_tokens | 282\.0 | tokens | No errors reported |
| 163553242cbd42f88de528586ea41141 | codex\_agent | baseline | cached\_tokens | 5666\.0 | tokens | No errors reported |
| 163553242cbd42f88de528586ea41141 | codex\_agent | baseline | reasoning\_tokens | 31\.0 | tokens | No errors reported |
| 81d2b34cf2ec43138224dc15b9c8c8f3 | claude\_code\_agent | skill | input\_tokens | 65174\.0 | tokens | No errors reported |
| 81d2b34cf2ec43138224dc15b9c8c8f3 | claude\_code\_agent | skill | output\_tokens | 614\.0 | tokens | No errors reported |
| 81d2b34cf2ec43138224dc15b9c8c8f3 | claude\_code\_agent | skill | cached\_tokens | 54802\.0 | tokens | No errors reported |
| 905b1d3739f643659fce8a2d178047e8 | claude\_code\_agent | baseline | input\_tokens | 87924\.0 | tokens | No errors reported |
| 905b1d3739f643659fce8a2d178047e8 | claude\_code\_agent | baseline | output\_tokens | 946\.0 | tokens | No errors reported |
| 905b1d3739f643659fce8a2d178047e8 | claude\_code\_agent | baseline | cached\_tokens | 76705\.0 | tokens | No errors reported |
| 24df69526bd94e1abe9ef73398e213d5 | codex\_agent | skill | input\_tokens | 18080\.0 | tokens | No errors reported |
| 24df69526bd94e1abe9ef73398e213d5 | codex\_agent | skill | output\_tokens | 201\.0 | tokens | No errors reported |
| 24df69526bd94e1abe9ef73398e213d5 | codex\_agent | skill | cached\_tokens | 11898\.0 | tokens | No errors reported |
| 24df69526bd94e1abe9ef73398e213d5 | codex\_agent | skill | reasoning\_tokens | 0\.0 | tokens | No errors reported |
| c7d99bb960fc4222800bb82c471f6e43 | codex\_agent | baseline | input\_tokens | 17606\.0 | tokens | No errors reported |
| c7d99bb960fc4222800bb82c471f6e43 | codex\_agent | baseline | output\_tokens | 240\.0 | tokens | No errors reported |
| c7d99bb960fc4222800bb82c471f6e43 | codex\_agent | baseline | cached\_tokens | 11545\.0 | tokens | No errors reported |
| c7d99bb960fc4222800bb82c471f6e43 | codex\_agent | baseline | reasoning\_tokens | 0\.0 | tokens | No errors reported |
| 103f9122477b4805b069e0ef7039e2a3 | claude\_code\_agent | skill | input\_tokens | 259730\.0 | tokens | No errors reported |
| 103f9122477b4805b069e0ef7039e2a3 | claude\_code\_agent | skill | output\_tokens | 1930\.0 | tokens | No errors reported |
| 103f9122477b4805b069e0ef7039e2a3 | claude\_code\_agent | skill | cached\_tokens | 245521\.0 | tokens | No errors reported |
| c3b24aa4e32145ffa47d260fe0c79716 | claude\_code\_agent | baseline | input\_tokens | 132013\.0 | tokens | No errors reported |
| c3b24aa4e32145ffa47d260fe0c79716 | claude\_code\_agent | baseline | output\_tokens | 787\.0 | tokens | No errors reported |
| c3b24aa4e32145ffa47d260fe0c79716 | claude\_code\_agent | baseline | cached\_tokens | 120794\.0 | tokens | No errors reported |
| 7e060361a8464ae19d09040c3af207f8 | codex\_agent | skill | input\_tokens | 18226\.0 | tokens | No errors reported |
| 7e060361a8464ae19d09040c3af207f8 | codex\_agent | skill | output\_tokens | 301\.0 | tokens | No errors reported |
| 7e060361a8464ae19d09040c3af207f8 | codex\_agent | skill | cached\_tokens | 11946\.0 | tokens | No errors reported |
| 7e060361a8464ae19d09040c3af207f8 | codex\_agent | skill | reasoning\_tokens | 19\.0 | tokens | No errors reported |
| 4cdb707cfac14e14b3e431e126647579 | codex\_agent | baseline | input\_tokens | 17603\.0 | tokens | No errors reported |
| 4cdb707cfac14e14b3e431e126647579 | codex\_agent | baseline | output\_tokens | 282\.0 | tokens | No errors reported |
| 4cdb707cfac14e14b3e431e126647579 | codex\_agent | baseline | cached\_tokens | 11532\.0 | tokens | No errors reported |
| 4cdb707cfac14e14b3e431e126647579 | codex\_agent | baseline | reasoning\_tokens | 0\.0 | tokens | No errors reported |
| a62848a7fba649b9b9c80b02ab0e884e | claude\_code\_agent | skill | input\_tokens | 65180\.0 | tokens | No errors reported |
| a62848a7fba649b9b9c80b02ab0e884e | claude\_code\_agent | skill | output\_tokens | 553\.0 | tokens | No errors reported |
| a62848a7fba649b9b9c80b02ab0e884e | claude\_code\_agent | skill | cached\_tokens | 54866\.0 | tokens | No errors reported |
| 37c9f932947a499ea08a03ea7f780ba7 | claude\_code\_agent | baseline | input\_tokens | 42781\.0 | tokens | No errors reported |
| 37c9f932947a499ea08a03ea7f780ba7 | claude\_code\_agent | baseline | output\_tokens | 411\.0 | tokens | No errors reported |
| 37c9f932947a499ea08a03ea7f780ba7 | claude\_code\_agent | baseline | cached\_tokens | 33014\.0 | tokens | No errors reported |
| 1384637df18f49e39c43d99b9c10d0f6 | codex\_agent | skill | input\_tokens | 56815\.0 | tokens | No errors reported |
| 1384637df18f49e39c43d99b9c10d0f6 | codex\_agent | skill | output\_tokens | 1002\.0 | tokens | No errors reported |
| 1384637df18f49e39c43d99b9c10d0f6 | codex\_agent | skill | cached\_tokens | 41727\.0 | tokens | No errors reported |
| 1384637df18f49e39c43d99b9c10d0f6 | codex\_agent | skill | reasoning\_tokens | 23\.0 | tokens | No errors reported |
| d65828e4d5a54f91bdc46f3579a01a41 | codex\_agent | baseline | input\_tokens | 79494\.0 | tokens | No errors reported |
| d65828e4d5a54f91bdc46f3579a01a41 | codex\_agent | baseline | output\_tokens | 1050\.0 | tokens | No errors reported |
| d65828e4d5a54f91bdc46f3579a01a41 | codex\_agent | baseline | cached\_tokens | 51054\.0 | tokens | No errors reported |
| d65828e4d5a54f91bdc46f3579a01a41 | codex\_agent | baseline | reasoning\_tokens | 32\.0 | tokens | No errors reported |
| 07d0c43f18e147eeb649254dac17c696 | claude\_code\_agent | skill | input\_tokens | 130219\.0 | tokens | No errors reported |
| 07d0c43f18e147eeb649254dac17c696 | claude\_code\_agent | skill | output\_tokens | 5025\.0 | tokens | No errors reported |
| 07d0c43f18e147eeb649254dac17c696 | claude\_code\_agent | skill | cached\_tokens | 106036\.0 | tokens | No errors reported |
| eadb036aaace4dbf99fe768036f95745 | claude\_code\_agent | baseline | input\_tokens | 177792\.0 | tokens | No errors reported |
| eadb036aaace4dbf99fe768036f95745 | claude\_code\_agent | baseline | output\_tokens | 1819\.0 | tokens | No errors reported |
| eadb036aaace4dbf99fe768036f95745 | claude\_code\_agent | baseline | cached\_tokens | 165924\.0 | tokens | No errors reported |
| 7afec0521bb04c13934127992153dc67 | codex\_agent | skill | input\_tokens | 130719\.0 | tokens | No errors reported |
| 7afec0521bb04c13934127992153dc67 | codex\_agent | skill | output\_tokens | 1059\.0 | tokens | No errors reported |
| 7afec0521bb04c13934127992153dc67 | codex\_agent | skill | cached\_tokens | 104497\.0 | tokens | No errors reported |
| 7afec0521bb04c13934127992153dc67 | codex\_agent | skill | reasoning\_tokens | 49\.0 | tokens | No errors reported |
| 730eaf95ab7d46b7b836bafb18aa615f | codex\_agent | baseline | input\_tokens | 71839\.0 | tokens | No errors reported |
| 730eaf95ab7d46b7b836bafb18aa615f | codex\_agent | baseline | output\_tokens | 948\.0 | tokens | No errors reported |
| 730eaf95ab7d46b7b836bafb18aa615f | codex\_agent | baseline | cached\_tokens | 56848\.0 | tokens | No errors reported |
| 730eaf95ab7d46b7b836bafb18aa615f | codex\_agent | baseline | reasoning\_tokens | 239\.0 | tokens | No errors reported |
| b20300c6cebe46e581d3032dde2611b5 | claude\_code\_agent | skill | input\_tokens | 128573\.0 | tokens | No errors reported |
| b20300c6cebe46e581d3032dde2611b5 | claude\_code\_agent | skill | output\_tokens | 2211\.0 | tokens | No errors reported |
| b20300c6cebe46e581d3032dde2611b5 | claude\_code\_agent | skill | cached\_tokens | 107255\.0 | tokens | No errors reported |
| 993c78e2cf0143eaaa2b653697924868 | claude\_code\_agent | baseline | input\_tokens | 42864\.0 | tokens | No errors reported |
| 993c78e2cf0143eaaa2b653697924868 | claude\_code\_agent | baseline | output\_tokens | 542\.0 | tokens | No errors reported |
| 993c78e2cf0143eaaa2b653697924868 | claude\_code\_agent | baseline | cached\_tokens | 32999\.0 | tokens | No errors reported |
| a5d89ee468264639b729053776b9548f | codex\_agent | skill | input\_tokens | 54133\.0 | tokens | No errors reported |
| a5d89ee468264639b729053776b9548f | codex\_agent | skill | output\_tokens | 540\.0 | tokens | No errors reported |
| a5d89ee468264639b729053776b9548f | codex\_agent | skill | cached\_tokens | 41639\.0 | tokens | No errors reported |
| a5d89ee468264639b729053776b9548f | codex\_agent | skill | reasoning\_tokens | 0\.0 | tokens | No errors reported |
| 2cc2f3794eb54fff95670c565f3a3487 | codex\_agent | baseline | input\_tokens | 34416\.0 | tokens | No errors reported |
| 2cc2f3794eb54fff95670c565f3a3487 | codex\_agent | baseline | output\_tokens | 677\.0 | tokens | No errors reported |
| 2cc2f3794eb54fff95670c565f3a3487 | codex\_agent | baseline | cached\_tokens | 20017\.0 | tokens | No errors reported |
| 2cc2f3794eb54fff95670c565f3a3487 | codex\_agent | baseline | reasoning\_tokens | 29\.0 | tokens | No errors reported |
| c6491df6d0c248ff9a8a99107b3f1e59 | claude\_code\_agent | skill | input\_tokens | 128904\.0 | tokens | No errors reported |
| c6491df6d0c248ff9a8a99107b3f1e59 | claude\_code\_agent | skill | output\_tokens | 3563\.0 | tokens | No errors reported |
| c6491df6d0c248ff9a8a99107b3f1e59 | claude\_code\_agent | skill | cached\_tokens | 106084\.0 | tokens | No errors reported |
| 65552a85f03f49d992bd5557c949fb87 | claude\_code\_agent | baseline | input\_tokens | 133127\.0 | tokens | No errors reported |
| 65552a85f03f49d992bd5557c949fb87 | claude\_code\_agent | baseline | output\_tokens | 1333\.0 | tokens | No errors reported |
| 65552a85f03f49d992bd5557c949fb87 | claude\_code\_agent | baseline | cached\_tokens | 122041\.0 | tokens | No errors reported |

</details>

## Known limits

Self-assessed evidence describes the recorded tasks and agents; it is not independent certification.

Signature: not verified. Signing and signature verification occur after artifact generation.

### Recorded evaluation scope

| Target | Evaluated surface | Bundled skills | Not evaluated |
| --- | --- | --- | --- |
| plugins/nvidia\-skills | BUNDLED SKILLS ONLY | \[&quot;nvidia\-skill\-finder&quot;\] | \[&quot;displayName&quot;,&quot;native\_plugin\_loading&quot;\] |

Scope is reported by the producer. Bundled-skill evaluation does not install or activate the plugin manifest, hooks, commands, MCP servers or other plugin components.

<details>
<summary>Methodology and provenance</summary>

| Field | Recorded value |
| --- | --- |
| Quality policy | nvcarps\-five\-dimensions\-v6 |
| Lift policy | nvcarps\-best\-agent\-baseline\-lift\-v2 |
| Expected attempts | 72 |
| Result SHA\-256 | f0bf005d42c2c34aca87238782cc21f5a08121d90afab66bab91685d2216add4 |

| Field | Supplied CI value |
| --- | --- |
| Content | plugins/nvidia\-skills |
| Source revision | b429a1dc269e665362399994a281be84bc904efd |
| Validation SHA\-256 | 41173f29d88c52b946a9562279ec1d687363eeb6f9d8b388c1235b04e898b41d |
| Gate SHA\-256 | 7716a3ab3c7518555c8b6eabf101f064944f0e90f85848a7ac9d3d2ead6b602a |

Original evidence is retained unchanged. Hashes establish correspondence, not producer authenticity.

</details>
