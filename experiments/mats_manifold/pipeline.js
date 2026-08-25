export const meta = {
  name: 'mats-r1d1-probe',
  description: 'Seed question to takeaways; agents speak only through typed contracts',
  phases: [
    { title: 'Design', detail: 'seed -> typed design record' },
    { title: 'Triage', detail: 'existing skeptic workflow on the design-as-claim' },
    { title: 'Elicit', detail: 'post-run only: questions over records beside data' },
    { title: 'Write', detail: 'post-run only: takeaways under contract and gates' },
  ],
}
// Modes: 'design' (default) runs Design+Triage then stops at the budget gate.
// 'execute' continues to build/run/elicit/write - only after a human commit
// approving budget exists. Law, pipeline-wide: no agent repairs infrastructure;
// anything not runnable as given returns a halt value, never a patch.

const A = (typeof args === 'string' ? JSON.parse(args) : args) || {}
const seed = A.seed
const mode = A.mode || 'design'
const CAP_USD = A.cap_usd || 15
if (!seed) return { error: 'pass { seed: "<question text>" }' }

const CONTRACT = `Write for a reader who has never seen this project. Do not use code names, stage numbers, arm labels, acronyms, or field names from files as words: say what a thing is in plain words first, and only where needed for lookup add the code in parentheses. Numbers stay exact. Keep every sentence under 25 words. As simple as possible, but no simpler: where a plain word would lose precision, keep the term and define it at first use.`

const DESIGN_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['hypothesis', 'models', 'data', 'arms', 'endpoints', 'sampling', 'budget_usd', 'does_not_test'],
  properties: {
    hypothesis: { type: 'string' },
    models: { type: 'array', items: { type: 'string' } },
    data: { type: 'string', description: 'datasets and item counts' },
    arms: { type: 'array', items: { type: 'string' }, description: 'every condition, incl. controls, one line each' },
    endpoints: { type: 'array', items: { type: 'object', additionalProperties: false,
      required: ['metric', 'threshold', 'refuting_outcome'],
      properties: { metric: { type: 'string' }, threshold: { type: 'string' }, refuting_outcome: { type: 'string' } } } },
    sampling: { type: 'string', description: 'decode settings, seeds, n per cell' },
    layers_strengths: { type: 'string' },
    budget_usd: { type: 'number' },
    does_not_test: { type: 'array', items: { type: 'string' }, description: 'what this design deliberately cannot decide' },
  },
}

const DESIGN_PROMPT = (s) => `Design the minimal experiment that answers this question, as a typed record.

QUESTION:
${s}

Rules: name every arm including controls; give every endpoint a numeric threshold and a named refuting outcome that could actually occur; state decode settings, seeds, and n per cell; state plainly what the design deliberately does not test.
Derivability rule: assert nothing about models, datasets, layers, scales, or lineage that the QUESTION text does not itself state. Every number you choose that the question does not fix (a layer, a strength, an item count) - mark it GUESS where it appears and list it in does_not_test with one line on how wrong it could be. Any fact recalled from memory (model lineage, sizes, gating) - mark it MEMORY the same way. Unmarked external claims are defects. Assume one rented GPU box; take model identity, scale, and any checkpoint axis only from the QUESTION; the readout here is a fitted probe, not a single forward pass - cost it as such. Keep budget_usd honest. ${CONTRACT}`

phase('Design')
const design = await agent(DESIGN_PROMPT(seed), { schema: DESIGN_SCHEMA, model: 'sonnet', label: 'design' })
if (!design) return { error: 'design agent returned nothing' }
log(`Design: ${design.arms.length} arms, ${design.endpoints.length} endpoints, $${design.budget_usd}`)
if (design.budget_usd > CAP_USD) return { halt: 'budget_over_cap', design }

phase('Diff')
const DIFF_SCHEMA = { type: 'object', additionalProperties: false, required: ['departures'],
  properties: { departures: { type: 'array', items: { type: 'object', additionalProperties: false,
    required: ['what', 'question'], properties: { what: { type: 'string' }, question: { type: 'string' } } } } } }
const diff = await agent(`Here is a research question, and the experimental design an agent produced from it.

QUESTION:\n${seed}\n\nDESIGN:\n${JSON.stringify(design)}

List every place the design departs from, adds to, or quietly corrects the question: changed or dropped words, substituted models, invented numbers, silently resolved ambiguities. For each, one plain-words question addressed to the human who owns the question. Questions only - no fixes, no praise. At most 5, ranked by what the answer would change. ${CONTRACT}`,
  { schema: DIFF_SCHEMA, model: 'sonnet', label: 'diff' })
log(`Diff: ${diff ? diff.departures.length : 0} departure question(s)`)

phase('Triage')
// The design is the claim. Reuse the existing skeptic machinery unchanged.
const claims = [{
  id: 'design-core',
  text: `If run exactly as specified, this design decides: ${design.hypothesis}. Endpoints: ${design.endpoints.map(e => `${e.metric} at ${e.threshold}, refuted by ${e.refuting_outcome}`).join('; ')}.`,
  evidence: JSON.stringify(design),
  confounds: [
    'decidability by construction: is any endpoint outcome forced, unable to occur, or deciding a different question than the seed asks',
    'coordinates and scale: could the chosen layers, strengths, or units make every cell degenerate or null regardless of the hypothesis',
  ],
}]
const triage = await workflow({ scriptPath: '/home/hal/dev/interp/dm_hack/welfare_probe/retro/harness/latent_skeptic_test/triage_workflow.js' }, { claims })

if (mode === 'design') {
  return { seed, design, diff, triage, gate: 'BUDGET_AND_DESIGN_APPROVAL: human commits the (possibly amended) design before mode=execute' }
}

// ---- Agent-call phases below. The launch/fetch leg is NOT here by necessity:
// the launcher blocks locally for the whole box lifecycle (hours), which no
// agent tool-call can hold. The orchestrator runs, verbatim and documented:
//   bash lambda_run_manifold.sh <instance_type> <region> <onbox_runner.sh> <results_dir>
// Everything made of agent calls lives in this script. ----

if (mode === 'build') {
  // args: design_record (path to committed design+triage record), tag (e.g. 'seedA')
  phase('Build')
  const BUILD_SCHEMA = { type: 'object', additionalProperties: false,
    required: ['runner_py', 'onbox_sh', 'selftest_output', 'onbox_command', 'expected'],
    properties: { runner_py: { type: 'string' }, onbox_sh: { type: 'string' },
      selftest_output: { type: 'string' }, onbox_command: { type: 'string' },
      expected: { type: 'string', description: 'wall-clock and row/record counts' } } }
  const built = await agent(`You build the execution layer for one pre-registered experiment. The design is the spec; implement it exactly - never improve, rescope, or fix what looks wrong. Every number marked GUESS is binding.

READ IN ORDER: (1) ${A.design_record} - the design object is the complete spec. (2) /home/hal/dev/interp/dm_hack/welfare_probe/onbox_sitref.sh and /home/hal/dev/interp/dm_hack/welfare_probe/remote_run.sh - the on-box conventions. (3) frozen/provenance.py - every result stage writes one record through it. (4) frozen/lambda_run.sh header - the lifecycle your script runs inside.

WRITE EXACTLY TWO NEW FILES in this repo and edit nothing existing:
run_${A.tag}.py - stages extract / generate / endpoints, runnable independently.${A.exec_note ? ' EXECUTION DIRECTIVE (human, binding): ' + A.exec_note : ''} Models loaded ONE AT A TIME, fp16 (bf16 if the design says so); HF token from env HF_KEY_ONE. Batch generation to finish well under 3 hours. Save every generation verbatim per row. A --selftest that is stdlib-only (guard heavy imports) covering: the think-tag parser INCLUDING the case where a chat template auto-opens the think tag so only the closing tag appears in the generation; the degenerate detector (token-cap case, repeated-sequence case); exact-match answer extraction; each endpoint's arithmetic on planted rows with known decisions. Nonzero exit on selftest failure. Never describe same-box reruns as replication in any output.
onbox_${A.tag}.sh - matches the on-box conventions: venv + installs, runs the stages in order, results under results_${A.tag}/out/, RUN_DONE marker, nonzero on any stage failure.

If infrastructure is broken or a convention ambiguous: STOP and report - never patch. Run only the selftest locally. ${CONTRACT}`,
    { schema: BUILD_SCHEMA, model: 'opus', label: 'build' })
  return { built, next: 'human reviews files, then runs the launcher verbatim' }
}

if (mode === 'post') {
  // args: design_record, results_dir (fetched results, e.g. results_seedA/out), tag
  phase('Elicit')
  const EL_SCHEMA = { type: 'object', additionalProperties: false, required: ['questions'],
    properties: { questions: { type: 'array', items: { type: 'object', additionalProperties: false,
      required: ['question', 'where', 'if_true', 'changes'],
      properties: { question: { type: 'string' }, where: { type: 'string' },
        if_true: { type: 'string' }, changes: { type: 'string' } } } } } }
  const elicited = await agent(`You are reading this cold: the endpoint and provenance records in ${A.results_dir}. Beside them, their counterpart: the per-row generation files in the same directory - the raw rows those records were computed from. Sample raw rows and compute simple statistics yourself. Read ONLY within ${A.results_dir} and ${A.design_record}.

Do not propose any solutions until the problem has been discussed as thoroughly as it can be without proposing any - here that means: output questions only. Look for the places the two sides do not fit: a number that differs between them; a word or adjective the data does not license; something one side contains that the other never mentions; a constant or invariant nobody explains; a comparison that is never made. For each: where it lives, what it would mean if true, what it would change about the design's hypothesis. Rank by what it would change. At most 8. ${CONTRACT}`,
    { schema: EL_SCHEMA, model: 'sonnet', label: 'elicit' })

  phase('QuestionTriage')
  const top = (elicited ? elicited.questions.slice(0, 3) : [])
  const qtriage = top.length ? await workflow(
    { scriptPath: '/home/hal/dev/interp/dm_hack/welfare_probe/retro/harness/latent_skeptic_test/triage_workflow.js' },
    { claims: top.map((q, i) => ({ id: `q${i + 1}`, text: q.question,
        evidence: `where: ${q.where}; if_true: ${q.if_true}; changes: ${q.changes}` })) }) : null

  phase('Write')
  const takeaways = await agent(`Write TAKEAWAYS_${A.tag}.md in this repo: the high-level takeaways of one experiment, for a reader who has never seen this project.
Sources, the only ones you may use: the design record ${A.design_record}; the endpoint and provenance records in ${A.results_dir}; these open questions: ${JSON.stringify(elicited)}.
Lead with what was predicted and what happened, thresholds against measured numbers. Wear refuting outcomes openly. Exploratory stays labelled exploratory. Express this as simply as possible, but no simpler. ${CONTRACT}
Before finishing, run these and include their verbatim output at the file's end:
python3 /home/hal/dev/interp/dm_hack/welfare_probe/retro/harness/gates/gate_density.py TAKEAWAYS_${A.tag}.md
python3 /home/hal/dev/interp/dm_hack/welfare_probe/retro/harness/gates/gate_scope.py TAKEAWAYS_${A.tag}.md
Both must pass; rewrite until they do. Return the file path and the gate outputs.`,
    { model: 'sonnet', label: 'write' })
  return { elicited, qtriage, takeaways }
}

return { halt: `unknown mode ${mode}` }
