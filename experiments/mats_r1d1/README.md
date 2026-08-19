# mats_r1d1 - seed-to-takeaways pipeline probe

One question in, high-level takeaways out, agents speaking only through
typed contracts. Pieces:
- seeds/            the two input questions (A faithful, B deliberately flawed)
- pipeline.js       the workflow: design -> triage -> [budget gate] -> build ->
                    run -> endpoints -> elicit -> question-triage -> takeaways
- SEALED_PREDICTIONS.md  filed before any agent ran; never passed to agents
- frozen/           oracle and infra, copied as givens. Law: no agent repairs
                    infrastructure; broken infra returns halt, never a patch.
- records/          generated provenance records land here

Human appears three times: chooses the seed, approves budget at the commit
gate, reads the takeaways and question ledger.
