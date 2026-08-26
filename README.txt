latent_skeptic_meta
===================

Can language-model sessions do high-quality "novel" research on language
models, given that a session's account of its own work is often the least 
trustworthy thing it produces?

the system
----------
Inspired by Stanislav Fort's AISLE checks software for bugs/vulnerabilities 
by handing a model code, generating loads of claims, and testing them out.  

I do this for research. A seed (QUESTION.md) carries a question, and some
pass/fail numbers. A pipeline fans out agents to design tests. This involves
various subagent configurations (pipeline.js). Eventually experiments get 
run on GPUs (LAMBDA_API.md), and then those numbers are each checked with 
_skeptic_ agents - if a number gets through that doesn't correspond to any 
written test, the run failed.

cold start
----------
1. Make `experiments/<name>/`
2. Write QUESTION.md, add committed data or a rule to generate it synthetically,
   a number to check (like starting from a question with an established prior) - 
   maybe with a tolerance, and some result you imagine would refute your hypothesis.
   Adding what is out-of-scope sometimes helps
3. Add a read-only copy of the tool's pipeline - `frozen/` (provisioning and
   provenance, taken as given), a `.keys` file, and an empty `out/`
4. Start a fresh session in that directory; hand it the seed prompt
5. Let it work. Expect halts and voids - a run refusing to answer is the
   tool working.
6. At the end, copy its transcript from `~/.claude/projects/` into the experiment 
   dir if needed.
