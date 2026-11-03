# ADR 011: Governed neural weight evolution

Status: accepted for this research implementation. Production activation and recursive self improvement remain outside demonstrated scope.

## Decision

RuVLLM owns mechanical model inspection, bounded patching, reversible checkpoint changes and evidence packaging. An external authority owns promotion and activation. A proposer cannot turn its own benchmark or signature into production permission.

The implementation separates three experimental claims:

1. Self repair corrects a specific observed failure mechanism.
2. Self optimization improves previously unseen tasks under an unchanged evaluation protocol.
3. Recursive self improvement makes the optimizer measurably better at producing subsequent improvements at equal compute.

Each claim requires separate evidence. Reproducing an existing published edit does not establish an independently discovered repair or recursive improvement.

## Checkable invariants

1. The source checkpoint remains unchanged. Candidate and rollback outputs use new paths.
2. Every changed tensor and value must fit an explicit policy. Protected ranges, changed value count and absolute or relative delta bounds are checked before publication.
3. A candidate is bound to its original checkpoint, patch, policy and generation identities.
4. Paired comparisons use the same prompts, decoding settings, output budgets and serving precision. Completion and delivered correctness are separate measurements.
5. Development data select candidates. Release evidence is retained, including regressions and failed attempts.
6. Evaluation validates model, runtime, protocol and topology identities before aggregation. Mixed generations are rejected.
7. Qualification, authorization and observed activation are distinct records. This repository exposes no production activation tool.
8. Rollback restores original bytes, verified by full digest and byte comparison.
9. A deployment generation change requires new cache identity across every serving peer. Distributed peer enforcement remains an external integration requirement.

## Implemented boundary

The Rust engine and scoped plugin implement inspection, bounded edits, rollback, receipt validation and RVF integrity. The Python runners implement actual inference comparison and independent saved output scoring. The coding adapter and surgical edit are evaluated separately.

Core Memory authority was tested in a separate local reference workspace using proposer, evaluator and readback credentials under one operator. This repository does not include that service or claim independently operated certification. The measured 0.8B quality gate failed and did not authorize production activation.

MetaHarness, RuVector, AgentDB, MeshLLM and Cognitum define integration roles in the wider architecture. Their complete distributed lifecycle is not implemented by this standalone publication. Native pretrained WASM editing, automatic EWC protection for base weight edits and recursive optimizer improvement are not demonstrated.

## Consequences

The mechanism can produce useful scientific evidence even when a candidate fails. In the recorded 0.8B experiment, tokens fell while delivered correctness regressed. Keeping the quality gate intact prevented that efficiency result from becoming a false improvement claim.

The cost is additional storage, hashing, evaluation and authority bookkeeping. Full checkpoint verification and exact rollback are deliberate costs. A future faster path must preserve the same externally checkable identities and restoration guarantees.

## Acceptance

Run the native CPU smoke fixture, verify all completed benchmark records, and reproduce byte exact rollback. For a future production candidate, additionally require an independent quality evaluation, protected capability checks, explicit external authorization and a runtime acknowledgment bound to the exact accepted generation.
