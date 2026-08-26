latent_skeptic_meta
===================

Private, permanently. The session transcripts in this repo contain full
tool output, local paths, keys-adjacent commands, and infrastructure
detail.

Tool-under-use is latent_skeptic/ 

To answer "how was this done": ledger -> records -> transcript last. 

Raw results (prompts, generations, per-row records, vectors) are tracked.

Single files at git-LFS size (100MB+) are .gitignored (currently nine 13B 
activation dumps, 2.4GB, under auto_interp), rederivable cheaply with a GPU run.


Sessions
--------
Transcripts of sessions that drove one experiment sit inside that
experiment's directory. Transcripts that span the program sit in
experiments/meta/, indexed there (name, dates, sidecar usage, one line
of content each): experiments/meta/INDEX.txt.


New experiment
--------------
1. Make experiments/<name>/ with: 
	QUESTION.md (committed data or generator rule; anchor + band, or 
	"anchor absent" declared; refuting outcome per claim; one-cell scope; 
	declared construct risks), 
	pipeline.js (frozen copy from the latent_skeptic pin, chmod 444), 
	frozen/ (provenance.py, lambda_run.sh, remote_run.sh), 
	LAMBDA_API.md, 
	.keys symlink, 
	empty out/

2. Then, in that directory: 
	`tmux new-session -As <name>`, run Claude (--dangerously-skip-permissions):
	
	```
    	Your /goal: answer the research question in QUESTION.md in this
    	directory, even if the answer is a refutation of the question.
    	pipeline.js is a frozen claim-triage workflow: your only audit tool,
    	read-only; never edit it or work around its verdicts. 

	frozen/ is given infrastructure - adapt only at the points its file 
	headers declare; broken infrastructure means halt and report, never patch. 

	.keys holds Lambda and Hugging Face keys; LAMBDA_API.md explains provisioning.
    	
	Rules: total spend cap 15 USD; confirm every instance teardown via the
    	API; never edit a launcher while a box runs - copy per launch.
    	
	Pre-register your design, bands, and hashes in PREREG.md before any
    	run. Audit your claims with the pipeline before measuring and again
    	after, with identical claim framing. 

	Save everything under out/.
    	
	Think and explain yourself as simply as possible, but no simpler. Do
    	not provide solutions until you have thought and reasoned about the
    	problem as thoroughly as possible without providing any.
	```

Meaning
-------------
The prompt lets this "director" or "manager" agent pickup on the interface for 
the pipeline - the subagent prompts, how they are used, etc - directly from the
pipeline.js header. It loosely enforces the "frozen" nature of the harness, and
the read-only nature. This is essential to enforcing editorial independence when 
reviewing results and work. 

The prompt never explains the pipeline's interface; the session learns it
from pipeline.js's own header comment. Do not message the session while it
is mid-turn: a queued message can be silently clobbered. Expect and want
halts: a gated dataset, a failed anchor, or an unbuildable instrument is
the pipeline working. Afterwards: copy the session's transcript jsonl from
~/.claude/projects/ into the experiment directory and add it to Sessions
above.
