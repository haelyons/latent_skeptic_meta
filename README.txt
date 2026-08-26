latent_skeptic_meta
===================

Can language-model sessions do research on language models that deserves
to be believed - given that a session's account of its own work is the
least trustworthy thing it produces?

the system
----------
The design is borrowed. Stanislav Fort's AISLE - and the older open-source
version of his workflow - checks software by handing a plain model the
code and one claim at a time. I do that to research. A question file
carries its data and its pass/fail numbers. A frozen pipeline fans out
fresh, cheap model contexts: one re-derives every number from the raw
files, each skeptic gets a single doubt to press, and none of them sees
the story. A rented GPU runs the experiment for about a dollar.
Everything lands in files a later reader can recompute. The models are
plain; the arrangement does the work. If a wrong number gets through and
the record can't convict it, the system failed.

actually check
--------------
I stopped treating text as evidence after months of runs whose write-ups
read well and couldn't be checked. Now I count a result when its number
is committed where a later reader can recompute it, and a mechanism when
someone cut the wire and watched the light. I once described data to a
run instead of shipping it: the runner rebuilt the prompts from memory
and faithfully measured the wrong experiment. Shipping the exact prompts
brought the recorded number back to two decimals.

Every catch I have on record came from a second look landing fresh: a
reader re-deriving numbers from raw files, a skeptic handed one doubt to
press, a diff between what I asked and what got built. I also found that
instructions only bite when they name the thing - naming the banned
word-categories cut jargon twenty-fold; the principle alone moved
nothing, three runs straight.

cold start
----------
1. Make experiments/<name>/.
2. Write QUESTION.md: the question; its committed data, or the rule that
   generates it; the number to reproduce first, with its tolerance; what
   would refute each claim; what is out of scope.
3. Add a read-only copy of the tool's pipeline; frozen/ (provisioning and
   provenance, taken as given); the keys symlink; an empty out/.
4. Start a fresh session in that directory; hand it the seed prompt.
5. Let it work. Expect halts and voids - a run refusing to answer is the
   tool working.
6. Copy its transcript from ~/.claude/projects/ into the experiment dir.

Worked example: experiments/manifold_c, newest and smallest. To answer
"how was this done": takeaways, then ledger, then records, then
transcript.

Private, permanently - transcripts carry full tool output, local paths,
and infrastructure detail.
