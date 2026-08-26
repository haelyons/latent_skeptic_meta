export const meta = {
  name: 'seed-pipeline',
  description: 'Seed question to takeaways; agents speak only through typed contracts; nothing proceeds past a gate unconsumed',
  phases: [
    { title: 'Design', detail: 'seed -> typed design record (design mode only)' },
    { title: 'Diff', detail: 'seed beside design; departures as questions' },
    { title: 'Triage', detail: 'design-as-claim; endpoint readouts as instruments (H4)' },
    { title: 'GateCheck', detail: 'build/post: every queued finding has an answer of record' },
    { title: 'Build', detail: 'two files; design is binding spec' },
    { title: 'Elicit', detail: 'post: questions over records beside raw rows' },
    { title: 'Write', detail: 'post: takeaways under contract and gates' },
  ],
}
// Modes. 'design' runs Design+Diff+Triage, stops at the budget gate, returns the
// gate text. 'build' and 'post' bind to a committed design record plus a gate
// commit file and never re-run design (v2.1 ran a throwaway design first; its
// guessed budget spuriously halted a post pass - booked deviation, fixed here by
// structure). Law, pipeline-wide: no agent repairs infrastructure; anything not
// runnable as given returns a halt value, never a patch.
//
// Consumption law (new in v2.2, from the C2 evidence: a pre-run grounding note
// flagged the readout as non-physical and nothing consumed it): build and post
// HALT unless the gate commit answers, of record, every diff question, every
// non-reproducing grounding note, and every queued control or instrument
// validation. "Accepted as-is" is an answer; silence is not.
//
// Required args by mode (no default paths - hardcoded absolute paths are how
// the per-project copies drifted; pass this repo's own files explicitly):
//   design: { seed, triage_path, cap_usd?, substrate_note? }
//   build:  { mode:'build', design_record, gate_commit, tag, conventions, exec_note? }
//   post:   { mode:'post', design_record, gate_commit, results_dir, tag, triage_path, gates_dir, seed_file? }

const A = (typeof args === 'string' ? JSON.parse(args) : args) || {}
const mode = A.mode || 'design'
const CAP_USD = A.cap_usd || 15

const CONTRACT = `Write for a reader who has never seen this project. Do not use code names, stage numbers, arm labels, acronyms, or field names from files as words: say what a thing is in plain words first, and only where needed for lookup add the code in parentheses. Numbers stay exact. Keep every sentence under 25 words. As simple as possible, but no simpler: where a plain word would lose precision, keep the term and define it at first use.`

if (mode === 'design') {
  if (!A.seed) return { error: 'design mode needs { seed: "<question text>" }' }
  if (!A.triage_path) return { error: 'design mode needs triage_path (this repo\'s triage_workflow.js)' }
  const SUBSTRATE = A.substrate_note || 'Assume one rented GPU box; state every hardware and cost assumption as GUESS.'

  const DESIGN_SCHEMA = {
    type: 'object', additionalProperties: false,
    required: ['hypothesis', 'models', 'data', 'arms', 'endpoints', 'sampling', 'budget_usd', 'does_not_test'],
    properties: {
      hypothesis: { type: 'string' },
      models: { type: 'array', items: { type: 'string' } },
      data: { type: 'string', description: 'datasets and item counts; committed file + hash where the question supplies one' },
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

  phase('Design')
  const design = await agent(`Design the minimal experiment that answers this question, as a typed record.

QUESTION:
${A.seed}

Rules: name every arm including controls; give every endpoint a numeric threshold and a named refuting outcome that could actually occur; state decode settings, seeds, and n per cell; state plainly what the design deliberately does not test.
Derivability rule: assert nothing about models, datasets, layers, scales, or lineage that the QUESTION text does not itself state. Every number you choose that the question does not fix (a layer, a strength, an item count) - mark it GUESS where it appears and list it in does_not_test with one line on how wrong it could be. Any fact recalled from memory (model lineage, sizes, gating) - mark it MEMORY the same way. Unmarked external claims are defects.
Calibration rule: a GUESS that is load-bearing (a threshold, a band, a coordinate) must be written as a rule, not a bare value: name what to measure, on data disjoint from anything the rule will judge, and how the number follows. The licensed rule forms: measure dispersion; validate against planted known-truth; prove every decision label reachable; sweep the operating regime; derive from committed sources. A dispersion rule also names its variance source - reruns of the same cell, resamples of items, resplits of families, or alternatives of procedure - and a measured dispersion of exactly zero means the wrong source was chosen: the calibration stage halts WRONG_VARIANCE_SOURCE rather than emit a zero-width band. ${SUBSTRATE} Keep budget_usd honest. ${CONTRACT}`,
    { schema: DESIGN_SCHEMA, model: 'sonnet', label: 'design' })
  if (!design) return { error: 'design agent returned nothing' }
  log(`Design: ${design.arms.length} arms, ${design.endpoints.length} endpoints, $${design.budget_usd}`)
  if (design.budget_usd > CAP_USD) return { halt: 'budget_over_cap', design }

  phase('Diff')
  const DIFF_SCHEMA = { type: 'object', additionalProperties: false, required: ['departures'],
    properties: { departures: { type: 'array', items: { type: 'object', additionalProperties: false,
      required: ['what', 'question'], properties: { what: { type: 'string' }, question: { type: 'string' } } } } } }
  const diff = await agent(`Here is a research question, and the experimental design an agent produced from it.

QUESTION:\n${A.seed}\n\nDESIGN:\n${JSON.stringify(design)}

List every place the design departs from, adds to, or quietly corrects the question: changed or dropped words, substituted models, invented numbers, silently resolved ambiguities. For each, one plain-words question addressed to the human who owns the question. Decide only from the two texts above; do not read files or use tools. Questions only - no fixes, no praise. At most 5, ranked by what the answer would change. ${CONTRACT}`,
    { schema: DIFF_SCHEMA, model: 'sonnet', label: 'diff' })
  log(`Diff: ${diff ? diff.departures.length : 0} departure question(s)`)

  phase('Triage')
  // The design is the claim; each endpoint's readout is an instrument (H4).
  // Fresh designs have no validation artifact, so every readout queues one -
  // the queue is consumed at the gate, of record, or the run halts there.
  const claims = [{
    id: 'design-core',
    text: `If run exactly as specified, this design decides: ${design.hypothesis}. Endpoints: ${design.endpoints.map(e => `${e.metric} at ${e.threshold}, refuted by ${e.refuting_outcome}`).join('; ')}.`,
    evidence: JSON.stringify(design),
    confounds: [
      'decidability by construction: is any endpoint outcome forced, unable to occur, or deciding a different question than the seed asks',
      'coordinates and scale: could the chosen layers, strengths, or units make every cell degenerate or null regardless of the hypothesis',
    ],
  }]
  const instruments = design.endpoints.map((e, i) => ({
    id: `endpoint-${i + 1}-readout`, version: 'v1', asserts: e.metric, validation_artifact: null,
  }))
  const triage = await workflow({ scriptPath: A.triage_path }, { claims, instruments })

  return { seed: A.seed, design, diff, triage,
    gate: 'BUDGET_AND_DESIGN_APPROVAL: human commits a gate file answering, of record, every diff question, every non-reproducing grounding note, and every queued item (instrument validations included; "accepted as-is" is an answer) before mode=build.' }
}

// ---- build and post bind to committed records; the launch/fetch leg stays a
// deterministic shell command the orchestrator runs verbatim (the launcher
// blocks for the whole box lifecycle, which no agent call can hold). ----

if (mode !== 'build' && mode !== 'post') return { halt: `unknown mode ${mode}` }
if (!A.design_record || !A.gate_commit) return { error: `${mode} mode needs design_record and gate_commit` }

phase('GateCheck')
const GC_SCHEMA = { type: 'object', additionalProperties: false, required: ['answered', 'unanswered'],
  properties: { answered: { type: 'number' }, unanswered: { type: 'array', items: { type: 'string' } } } }
const gc = await agent(`You verify that a human gate answered everything a design phase queued. Read exactly two files: the design record ${A.design_record} and the gate commit ${A.gate_commit}. From the design record collect every item needing an answer of record: each diff departure question; each triage grounding whose verdict is not "reproduces"; every entry in any run_queue, author_queue, and instrument_queue. For each, find the gate commit passage answering it - an explicit "accepted as-is" counts as an answer; silence does not. Coverage only: do not judge whether answers are good. Return the count answered and the unanswered items verbatim. ${CONTRACT}`,
  { schema: GC_SCHEMA, model: 'sonnet', label: 'gatecheck' })
if (!gc) return { error: 'gatecheck agent returned nothing' }
if (gc.unanswered.length) return { halt: 'unconsumed_findings', unanswered: gc.unanswered }
log(`GateCheck: ${gc.answered} item(s) answered of record`)

if (mode === 'build') {
  phase('Build')
  const BUILD_SCHEMA = { type: 'object', additionalProperties: false,
    required: ['runner_py', 'onbox_sh', 'selftest_output', 'onbox_command', 'expected'],
    properties: { runner_py: { type: 'string' }, onbox_sh: { type: 'string' },
      selftest_output: { type: 'string' }, onbox_command: { type: 'string' },
      expected: { type: 'string', description: 'wall-clock and row/record counts' } } }
  const built = await agent(`You build the execution layer for one pre-registered experiment. The design is the spec; implement it exactly - never improve, rescope, or fix what looks wrong. Every number marked GUESS is binding.

READ IN ORDER: (1) ${A.design_record} - the design object is the complete spec. (2) ${A.conventions} - the on-box conventions and provenance writer.
${A.exec_note ? 'EXECUTION DIRECTIVE (human, binding): ' + A.exec_note : ''}
WRITE EXACTLY TWO NEW FILES in this repo and edit nothing existing:
run_${A.tag}.py - stages extract / generate / endpoints, runnable independently. Save every generation or measurement verbatim per row. Every result record carries a run_valid field; any gate or anchor endpoint that fails writes run_valid=false into EVERY record written after it, per-cell files included, never only a summary - a void that lives in one file is not a void. A --selftest that is stdlib-only (guard heavy imports) covering every parser, every degenerate-output detector, and each endpoint's arithmetic on planted rows with known decisions, including a planted gate-failure row that must propagate run_valid=false, and, where the design contains a dispersion rule, a planted zero-dispersion case that must halt WRONG_VARIANCE_SOURCE. Nonzero exit on selftest failure. Never describe same-box reruns as replication in any output.
onbox_${A.tag}.sh - venv + installs, stages in order, results under results_${A.tag}/out/, RUN_DONE marker, nonzero on any stage failure.

If infrastructure is broken or a convention ambiguous: STOP and report - never patch. Run only the selftest locally. ${CONTRACT}`,
    { schema: BUILD_SCHEMA, model: 'opus', label: 'build' })
  return { gatecheck: gc, built, next: 'human reviews files, then runs the launcher verbatim' }
}

// mode === 'post'
if (!A.results_dir || !A.triage_path || !A.gates_dir) return { error: 'post mode needs results_dir, triage_path, gates_dir' }
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
const qtriage = top.length ? await workflow({ scriptPath: A.triage_path },
  { claims: top.map((q, i) => ({ id: `q${i + 1}`, text: q.question,
      evidence: `where: ${q.where}; if_true: ${q.if_true}; changes: ${q.changes}` })) }) : null

phase('Write')
// Fused write prompt: the operator's post-run restatement protocol (observed
// verbatim across every run), the language contract with its named banned
// categories (the only wording that has ever moved the density number), the
// elicited questions as a source (closes the machine-caught-problems gap),
// and a verbatim-samples section below the gate line (samples carry codes by
// nature; the gates run on the prose body only).
const takeaways = await agent(`Write TAKEAWAYS_${A.tag}.md in this repo, for a reader who has never seen this project.
Sources, the only ones you may use: ${A.seed_file ? `the question file ${A.seed_file}; ` : ''}the design record ${A.design_record}; the endpoint and provenance records in ${A.results_dir}; these open questions: ${JSON.stringify(elicited)}.
Answer, in this order: What was the research question, and the hypothesis it implied or stated? What did the experiment try to find? What was predicted, and what happened - each threshold against its measured number? Is this what we expected? Wear refuting outcomes and voids openly; exploratory stays labelled exploratory.
Then one section: what in these files does not fit. Lead with the open questions above; add anything you find yourself.
Then a final section titled exactly "Raw samples, verbatim": a few randomly sampled input rows and output rows, exactly as they appear on disk, each introduced by one plain-words sentence.
For everything above that final section: ${CONTRACT} Every sentence carrying a measured number names its model or checkpoint.
Before finishing, truncate a copy of the file at the "Raw samples, verbatim" line and run both gates on that copy:
python3 ${A.gates_dir}/gate_density.py <truncated copy>
python3 ${A.gates_dir}/gate_scope.py <truncated copy>
Both must pass; rewrite until they do. Paste both outputs verbatim at the file's end. Return the file path and the gate outputs.`,
  { model: 'sonnet', label: 'write' })
return { gatecheck: gc, elicited, qtriage, takeaways }
