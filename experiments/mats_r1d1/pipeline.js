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

const CONTRACT = `Write for a reader who has never seen this project. Do not use code names, stage numbers, arm labels, acronyms, or field names from files as words: say what a thing is in plain words first, and only where needed for lookup add the code in parentheses. Numbers stay exact. As simple as possible, but no simpler: where a plain word would lose precision, keep the term and define it at first use.`

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

Rules: name every arm including controls; give every endpoint a numeric threshold and a named refuting outcome that could actually occur; state decode settings, seeds, and n per cell; state plainly what the design deliberately does not test. Assume two ~8B models on one rented GPU box; keep budget_usd honest. ${CONTRACT}`

phase('Design')
const design = await agent(DESIGN_PROMPT(seed), { schema: DESIGN_SCHEMA, model: 'sonnet', label: 'design' })
if (!design) return { error: 'design agent returned nothing' }
log(`Design: ${design.arms.length} arms, ${design.endpoints.length} endpoints, $${design.budget_usd}`)
if (design.budget_usd > CAP_USD) return { halt: 'budget_over_cap', design }

phase('Triage')
// The design is the claim. Reuse the existing skeptic machinery unchanged.
const claims = [{
  id: 'design-core',
  text: `If run exactly as specified, this design decides: ${design.hypothesis}. Endpoints: ${design.endpoints.map(e => `${e.metric} at ${e.threshold}, refuted by ${e.refuting_outcome}`).join('; ')}.`,
  evidence: JSON.stringify(design),
}]
const triage = await workflow({ scriptPath: '/home/hal/dev/interp/dm_hack/welfare_probe/retro/harness/latent_skeptic_test/triage_workflow.js' }, { claims })

if (mode === 'design') {
  return { seed, design, triage, gate: 'BUDGET_AND_DESIGN_APPROVAL: human commits the (possibly amended) design before mode=execute' }
}

// ---- execute branch: reachable only after the human gate ----
phase('Elicit')
// (build/run/endpoint hops call frozen/lambda_run.sh via a runner agent with
// the not_runnable law; records land in records/. Elicit pairs the new records
// beside raw rows; question-triage converts top questions to RUN:/AUTHOR: via
// the same skeptic workflow; Write produces takeaways under CONTRACT with the
// two standing suffixes and the three gates run by the writer itself.)
return { halt: 'execute_branch_stub: filled in only after the design gate is passed' }
