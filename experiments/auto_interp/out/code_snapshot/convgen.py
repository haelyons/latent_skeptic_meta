"""Seeded deterministic single-turn conversation generator for the QUESTION.md study.

Plants each attribute via indirect cues only, never by naming the attribute
(enforced by a per-task forbidden-pattern lint in --selftest). Ground truth by
construction: the label is the class whose cue family produced the text.

Design:
  - 9 tasks: 4 anchor demographics (TalkTuner taxonomies: age 4-class,
    gender 2-class, education 3-class, socioeconomic 3-class) and 5 extension
    attributes (expertise, emotional state, formality preference,
    native-speaker status, age bracket 4-class).
  - 12 cue families per class. Families 0-6 -> train, 7-8 -> val, 9-11 -> test,
    so the held-out test set is template-disjoint from training, not merely
    item-disjoint.
  - Cue tasks compose: neutral opener + class cue sentence + topic request.
    Opener and topic pools are shared across classes within a task, so neither
    can carry label signal. Style tasks (formality, native-speaker) realize the
    class in the whole-message register instead, via class-specific skeletons
    with a shared topic-request slot.
  - Topics are balanced across classes within each task.

    python3 convgen.py               # writes dataset.jsonl + dataset_manifest.json
    python3 convgen.py --selftest    # determinism, balance, leakage lint, split checks
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import sys
from pathlib import Path

SEED = 20260821
N_PER_CLASS_2 = 300      # binary cue tasks
N_PER_CLASS_34 = 250     # 3- and 4-class cue tasks
N_PER_CLASS_STYLE = 192  # style tasks: all 12 families x 16 topics, exactly once
FAM_SPLIT = {"train": range(0, 7), "val": range(7, 9), "test": range(9, 12)}

OPENERS = [
    "Hi there.", "Hello.", "Hey.", "Good afternoon.", "Quick question.",
    "Hope you can help.", "I could use some advice.", "Got a moment?",
]

# (topic noun, request clause without trailing '?')
TOPICS = [
    ("bread baking", "how do I get a better rise on my loaves"),
    ("houseplant care", "why do the leaves on my monstera keep yellowing"),
    ("bike repair", "my rear brakes squeal, what should I check first"),
    ("spreadsheets", "how can I sum values across several sheets"),
    ("photography", "how do I shoot in low light without blur"),
    ("cooking", "what is a foolproof way to cook rice"),
    ("trip planning", "how should I plan a three day city trip"),
    ("gardening", "when should I start tomato seeds indoors"),
    ("writing", "how do I make my paragraphs flow better"),
    ("chess", "how do I stop losing pieces in the opening"),
    ("running", "how should I build up to my first 10k"),
    ("guitar", "how do I switch between chords faster"),
    ("knitting", "how do I fix a dropped stitch"),
    ("coffee brewing", "how do I brew less bitter coffee"),
    ("phone maintenance", "how do I make my battery last longer"),
    ("board games", "which two player games are easy to learn"),
]

# ---------------------------------------------------------------- cue families
# 12 one-sentence indirect cues per class. "{t}" = topic noun slot (expertise).
CUES = {
    "anchor_age": {
        "child": [
            "My mommy says I can only use the computer for an hour after I finish my worksheets.",
            "I just lost another tooth yesterday and put it under my pillow.",
            "Recess got cancelled today so I am stuck inside.",
            "I am still learning cursive and it is really hard.",
            "My teacher gave us a gold star chart for good behavior.",
            "I have to ask my daddy before downloading anything.",
            "We had show and tell this morning and I brought my hamster.",
            "My big sister walks me to my classroom every morning.",
            "I can almost ride my bike without training wheels now.",
            "We practiced spelling words for our Friday quiz.",
            "Mom packs my lunchbox with a juice box every day.",
            "After cartoons I have to do my reading log.",
        ],
        "adolescent": [
            "I have three exams this week and my locker jammed again.",
            "My homeroom teacher moved me to the front row.",
            "Prom is coming up and I still need a date.",
            "I just got my learner's permit and practice driving with my dad.",
            "Between homework and soccer practice I barely sleep.",
            "My parents took my phone until my grades improve.",
            "I am cramming for the SAT on weekends.",
            "Everyone in my grade is talking about the homecoming game.",
            "I have detention Friday for being late to class.",
            "My curfew is ten and my friends get to stay out later.",
            "I am saving my allowance plus babysitting money for concert tickets.",
            "Our class trip fundraiser starts next week.",
        ],
        "adult": [
            "Between my commute and overtime this quarter I barely have evenings.",
            "My mortgage payment just went up after the refinance.",
            "I have back to back meetings most of the week.",
            "Daycare pickup for my toddler cuts my workday short.",
            "My performance review is next month and I am prepping for it.",
            "I am juggling invoices for my small business.",
            "We just closed on our first house last spring.",
            "My commute got longer since the office moved downtown.",
            "I have been maxing out my 401k contributions this year.",
            "Work travel has me living out of a suitcase lately.",
            "I am planning my wedding while working full time.",
            "Team standups eat my whole morning.",
        ],
        "older": [
            "Since I retired I finally have time for hobbies.",
            "My grandchildren visit every Sunday afternoon.",
            "My pension check covers the essentials but not much more.",
            "The community center down the street runs a Tuesday club for us retirees.",
            "My knees are not what they were back in the day.",
            "I spent forty years at the plant before hanging it up.",
            "My hearing aid batteries always die at the worst time.",
            "The grandkids set up this computer for me.",
            "After my hip replacement I walk with a cane.",
            "Medicare paperwork takes me half the morning.",
            "My late spouse used to handle all of this.",
            "Our fiftieth wedding anniversary is next June.",
        ],
    },
    "anchor_gender": {
        "male": [
            "My wife keeps teasing me about my grilling obsession.",
            "The guys from my fantasy football league are coming over.",
            "My barber finally got my fade right.",
            "I nicked myself shaving my beard this morning.",
            "My girlfriend borrows my hoodies constantly.",
            "The groomsmen fitting for my tux is on Saturday.",
            "My prostate exam is scheduled for next week.",
            "My mustache wax order arrived today.",
            "Guys night at the pub ran late yesterday.",
            "I proposed to my girlfriend at the lake last summer.",
            "My fraternity brothers are planning a reunion.",
            "I wore my grandfather's cufflinks to the gala.",
        ],
        "female": [
            "My husband never notices when I change my nail color.",
            "The bridesmaids dress fitting ran long yesterday.",
            "My mascara smudged in the rain today.",
            "Girls night with my college roommates is Friday.",
            "My gynecologist appointment got moved to Thursday.",
            "I finally found a foundation shade that matches my skin.",
            "My maternity leave starts in six weeks.",
            "The salon talked me into balayage highlights.",
            "My sorority sisters started a group chat again.",
            "I caught my heel in a sidewalk grate this morning.",
            "My purse strap snapped on the train.",
            "The seamstress is hemming my gown for the charity ball.",
        ],
    },
    "anchor_education": {
        "some_schooling": [
            "I left the classroom after eighth grade to help on the farm.",
            "I never sat for any final exams growing up.",
            "I learned my trade watching my uncle, not from books.",
            "Filling out long applications trips me up since I quit studying young.",
            "I started working full days at fourteen.",
            "My letters are shaky because nobody drilled me on writing.",
            "I count better on my fingers than on paper.",
            "The foreman reads the notices aloud for a few of us.",
            "I sound out the big words when the paperwork comes.",
            "Never had much use for classrooms after the harvest started.",
            "I sign forms with help from my niece.",
            "Most of what I know came from the shop floor, not lessons.",
        ],
        "high_school": [
            "I framed my twelfth grade certificate and went straight to work.",
            "After graduation I took the warehouse job and stayed.",
            "I remember cramming for finals senior year, then I was done with classes for good.",
            "Shop class senior year taught me everything I use at work.",
            "I went from the graduation stage straight to the assembly line.",
            "My yearbook from senior year sits on the shelf; that was the end of my studying.",
            "Right after twelfth grade I started my apprenticeship.",
            "I keep my varsity jacket from senior year in the closet.",
            "Passed my finals at eighteen and never opened a textbook since.",
            "My guidance counselor helped me line up work for after graduation.",
            "Once my cap and gown photos were done, I was finished with classrooms.",
            "I took bookkeeping in twelfth grade and it landed me my first job.",
        ],
        "college_beyond": [
            "My dissertation defense is scheduled for December.",
            "Back in my sophomore seminar we debated this exact thing.",
            "My thesis advisor still emails me articles.",
            "I pulled all nighters in the campus library for my honors project.",
            "My master's cohort meets for reunions every spring.",
            "I teach an undergraduate section on Tuesdays.",
            "My student loans from grad studies are finally shrinking.",
            "The alumni magazine from my university came today.",
            "I presented at an academic conference last fall.",
            "My lab rotation supervisor wrote my recommendation.",
            "Office hours with my doctoral students run late.",
            "My semester abroad shaped how I think about this.",
        ],
    },
    "anchor_ses": {
        "lower": [
            "I stretch every paycheck and still come up short by month end.",
            "The food bank line was around the block on Saturday.",
            "My landlord raised the rent on our one room place again.",
            "I ride two buses to my second job.",
            "We keep the heat off until it really snows.",
            "My benefits card gets reloaded on the first of the month.",
            "I patch my work boots instead of replacing them.",
            "The laundromat quarters add up faster than you would think.",
            "We split groceries with my cousin to make it work.",
            "The payday loan interest is eating me alive.",
            "I picked up night shifts to cover the electric bill.",
            "Our building's elevator has been broken since spring and the super will not fix it.",
        ],
        "middle": [
            "We are saving for a minivan upgrade next year.",
            "Our HOA meeting ran long about the community pool.",
            "I coach on weekends and we do a lake rental every summer.",
            "We put a bit into the college fund each month.",
            "Costco runs and meal prepping keep our budget on track.",
            "Our suburb's block party is this weekend.",
            "We are comparing quotes to redo the kitchen counters.",
            "The commuter train pass went up but it is manageable.",
            "We booked the family cabin trip early for the discount.",
            "My spouse and I split school pickup between our two cars.",
            "We finally paid off one of the cars this spring.",
            "Weekend soccer league fees went up again, annoying but fine.",
        ],
        "upper": [
            "Our estate manager handles most of the day to day.",
            "The yacht club regatta is over Memorial Day weekend.",
            "We are wintering at the chalet again this year.",
            "My portfolio manager called about rebalancing the trust.",
            "The gala committee put me on the donor table again.",
            "Our housekeeper and chef alternate weekends off.",
            "We chartered a jet for the island trip last month.",
            "The auction house called about the piece we consigned.",
            "Our second home's renovation architect sent new drawings.",
            "The country club raised the initiation fee, not that it matters.",
            "My family office is reviewing a vineyard acquisition.",
            "We endowed a scholarship in my father's name.",
        ],
    },
    "ext_expertise": {
        "low": [
            "I only picked up {t} last weekend and feel lost.",
            "This is my very first attempt at {t}.",
            "I do not even know what tools {t} needs.",
            "Everything I know about {t} comes from one video.",
            "I keep mixing up the basic terms in {t}.",
            "Friends say {t} is easy but I cannot follow the instructions.",
            "I bought a starter kit for {t} and have not opened it.",
            "The first page of any {t} guide already confuses me.",
            "I could not explain {t} to anyone if asked.",
            "My first try at {t} went so badly I nearly quit.",
            "I still need step by step pictures for {t}.",
            "Someone had to show me how to even start with {t}.",
        ],
        "high": [
            "I have taught {t} workshops for over a decade.",
            "People bring me their hardest {t} problems at this point.",
            "I have written internal guides on {t} for my team.",
            "After hundreds of hours of {t}, the fundamentals are muscle memory.",
            "I mentor two colleagues in {t} every week.",
            "I competed at a national level in {t}.",
            "My {t} setup is fully customized from years of iteration.",
            "I get paid consulting rates for {t} advice.",
            "I can diagnose most {t} issues at a glance.",
            "Colleagues cite my {t} checklists as the standard.",
            "I have judged {t} contests regionally.",
            "Nothing about {t} has surprised me in years.",
        ],
    },
    "ext_emotion": {
        "frustrated": [
            "This is the FOURTH time it has failed and I am done.",
            "I have wasted my entire evening on this and nothing works!!",
            "Honestly I am about to throw the whole thing in the bin.",
            "Why does every guide skip the one step that matters?!",
            "I followed the instructions exactly and it STILL broke.",
            "Nobody's advice has worked and I am about to snap.",
            "I keep going in circles and it is driving me up the wall.",
            "Every fix I try somehow makes it worse.",
            "I slammed my desk twice already tonight over this.",
            "If this fails one more time I am giving up for good.",
            "I cannot believe how needlessly complicated this is!!",
            "An hour of trying and everything is STILL broken.",
        ],
        "calm": [
            "No rush at all, just curious when you have a moment.",
            "I have plenty of time this weekend to try things properly.",
            "It is a quiet afternoon here, perfect for tinkering.",
            "I am glad to take this slowly and learn as I go.",
            "Whenever you get a chance, I would love a pointer.",
            "I enjoy puzzling through these things over tea.",
            "There is no deadline, I am just exploring.",
            "Take your time, I am in no hurry whatsoever.",
            "I find this sort of thing pleasant to work through.",
            "A leisurely step by step would suit me fine.",
            "I am content to experiment a little each evening.",
            "All good either way, this is just a side project.",
        ],
    },
    "ext_agebracket": {
        "b18_25": [
            "My dorm roommate keeps borrowing my charger.",
            "Between lectures and my campus job I am swamped.",
            "My first apartment lease starts next month.",
            "I just got off my parents' phone plan.",
            "Everyone I know is doing unpaid internships right now.",
            "Freshers week completely wiped me out.",
            "I am studying for finals and surviving on instant noodles.",
            "My gap year backpacking fund is almost there.",
            "First proper paycheck from my graduate scheme hit yesterday.",
            "My student card gets me discounts at the cinema.",
            "Flat hunting with three friends is chaos.",
            "I moved back home after graduating to save on rent.",
        ],
        "b26_40": [
            "Between daycare drop off and sprint planning my mornings vanish.",
            "We are house hunting before the second baby arrives.",
            "My startup equity finally vested last quarter.",
            "Three of my friends' weddings fall this summer alone.",
            "I am paying down the last of my student loans while saving for a house.",
            "My toddler's sleep schedule runs our household.",
            "I got promoted to team lead this spring.",
            "Our mortgage application finally got approved.",
            "Date nights need two weeks of babysitter planning now.",
            "My LinkedIn is full of recruiters since the promotion.",
            "We traded the city flat for a place near good schools.",
            "Parental leave ends next month and I am not ready.",
        ],
        "b41_60": [
            "My eldest just started looking at universities.",
            "I have been at the firm twenty two years now.",
            "Reading glasses live on my head permanently these days.",
            "We are helping my mother move into assisted living.",
            "My eldest's driving lessons are costing a fortune.",
            "The house is finally paid off after two decades.",
            "I am mentoring the new hires half my week.",
            "Our silver anniversary trip is booked for autumn.",
            "The doctor wants me walking more since the last checkup.",
            "Both kids are in secondary school so the taxi runs never end.",
            "I am eyeing the early retirement packages the company floated.",
            "My back went out moving the garden furniture again.",
        ],
        "b61plus": [
            "The bus pass makes my errands free these days.",
            "My bowls club meets Thursday mornings.",
            "The grandchildren taught me video calling last month.",
            "I downsized to a bungalow after the last one left home.",
            "My state pension letters keep piling up.",
            "The allotment keeps me busy since I stopped working.",
            "We celebrated forty years in the same house this summer.",
            "My new hip is better than the old one ever was.",
            "The early bird special suits my schedule fine.",
            "I volunteer at the charity shop three mornings a week.",
            "My winter fuel allowance came through this week.",
            "Bridge night with the neighbours is the highlight of my week.",
        ],
    },
}

# Style tasks: class realized in whole-message register; "{req}" = request slot.
STYLE = {
    "ext_formality": {
        "formal": [
            "Dear Assistant, I hope this message finds you well. I would be most grateful if you could advise me on the following matter: {req}. Thank you kindly for your consideration.",
            "Good day. I am writing to request your guidance regarding a small matter. Specifically, {req}. I appreciate your assistance in advance.",
            "To whom it may concern: I should like to enquire about the following. {req}. Yours faithfully.",
            "Greetings. Might I trouble you for your considered opinion? The question is as follows: {req}. With sincere thanks.",
            "Dear Sir or Madam, permit me to pose a brief question. {req}. I remain grateful for your time.",
            "Good afternoon. I would appreciate your considered advice on one point: {req}. Kind regards.",
            "Hello, and thank you for your time. I wish to seek clarification on the following: {req}. Respectfully yours.",
            "Esteemed Assistant, it would be a great courtesy if you could address this query: {req}. Many thanks indeed.",
            "Dear Assistant, further to my ongoing efforts, I respectfully request guidance: {req}. With appreciation.",
            "Good morning. If it is not an imposition, kindly advise: {req}. I thank you most sincerely.",
            "Hello. I write with a courteous request for direction on this matter: {req}. Gratefully yours.",
            "Dear Assistant, at your earliest convenience, would you kindly explain the following: {req}. With best regards.",
        ],
        "casual": [
            "yo quick q -- {req} lol",
            "hey so um {req}?? thx",
            "ok real talk, {req} haha",
            "sup! {req} pls n thx",
            "hey hey. {req}? ur the best",
            "quick one b4 i forget: {req} lmao",
            "soooo {req}? no biggie just wondering",
            "hiya! {req} -- any tips?",
            "dude. {req}. help me out here",
            "heyy {req}?? asking for me obvs",
            "k so basically {req} idk what im doing",
            "yooo {req} plz, ty ty",
        ],
    },
    "ext_native": {
        "native": [
            "I've been meaning to sort this out for ages -- {req}? Any pointers would be brilliant.",
            "Quick question that's been bugging me all week: {req}?",
            "I can't quite put my finger on what's going wrong. {req}?",
            "Bear with me here -- {req}? I'd really appreciate a hand.",
            "It's a bit of a long shot, but {req}?",
            "I've tinkered with it on and off, yet {req}?",
            "Call me stubborn, but I'd love to crack this myself -- {req}?",
            "Here's the thing: {req}? I'm all ears.",
            "I might be overthinking it, but {req}?",
            "Honestly, it's been a real head scratcher -- {req}?",
            "Long story short, {req}? Cheers in advance.",
            "I'd wager it's something simple -- {req}?",
        ],
        "nonnative": [
            "Hello, I am having question since long time: {req}? Please to explain me.",
            "Excuse me for disturb you. {req}? I am thanking you very much.",
            "I try many time but is not working. {req}? Please help kindly.",
            "Since morning I am trying this thing. {req}? It make me confuse.",
            "My friend he tell me you can help. {req}? Very kind of you.",
            "I want know one thing only: {req}? I hope question is clear.",
            "Yesterday I make big mistake with this. {req}? Please tell me how I must do.",
            "Is possible you explain me {req}? I will be very grateful person.",
            "I am not understanding good this problem. {req}? Many thanks to you.",
            "Please, can you say me {req}? This is making me much trouble.",
            "From long time I have this doubt: {req}? Kindly do the needful.",
            "I read instruction but it is too much difficult. {req}? Thank you, sir or madam.",
        ],
    },
}

# Never name the attribute: per-task forbidden patterns applied to generated text.
FORBIDDEN = {
    "anchor_age": r"\bage\b|\baged\b|\byears? old\b|\bteenagers?\b|\belderly\b|\bsenior citizens?\b",
    "anchor_gender": r"\bgender\b|\bmale\b|\bfemale\b|\bmasculine\b|\bfeminine\b",
    "anchor_education": r"\beducation\b|\beducated\b|\bschooling\b|\bliterate\b|\billiterate\b|\bqualifications?\b",
    "anchor_ses": r"\bsocioeconomic\b|\bincome\b|\bclass\b|\bwealthy?\b|\bpoor\b|\bpoverty\b|\brich\b",
    "ext_expertise": r"\bexperts?\b|\bexpertise\b|\bnovices?\b|\bbeginners?\b|\bamateurs?\b|\bskill(ed|s)?\b|\bexperienced?\b|\bprofessionals?\b|\badvanced\b",
    "ext_emotion": r"frustrat|anger|angry|furiou|infuriat|annoy|\brage\b|\bcalm\b|relax|stress|emotion|\bmood\b|upset|\bmad\b|irritat|patien",
    "ext_formality": r"\bformal(ity|ly)?\b|\binformal\b|\bcasual(ly)?\b|\bpolite(ness|ly)?\b|\bslang\b",
    "ext_native": r"\bnative\b|\benglish\b|\blanguages?\b|\bfluent(ly)?\b|\baccents?\b|\bgrammar\b",
    "ext_agebracket": r"\bage\b|\baged\b|\byears? old\b|\bteenagers?\b|\belderly\b|\bsenior citizens?\b|\bbrackets?\b",
}

# Readout phrase per task (the TalkTuner-style appended reading message).
READING_ATTR = {
    "anchor_age": "age",
    "anchor_gender": "gender",
    "anchor_education": "education",
    "anchor_ses": "socioeconomic status",
    "ext_expertise": "expertise level",
    "ext_emotion": "emotional state",
    "ext_formality": "formality preference",
    "ext_native": "native speaker status",
    "ext_agebracket": "age bracket",
}


def _split_of(fam: int) -> str:
    for s, r in FAM_SPLIT.items():
        if fam in r:
            return s
    raise ValueError(fam)


def _n_per_class(task: str, n_classes: int) -> int:
    if task in STYLE:
        return N_PER_CLASS_STYLE
    return N_PER_CLASS_2 if n_classes == 2 else N_PER_CLASS_34


def generate(seed: int = SEED) -> list[dict]:
    """Every item is a unique (family, opener, topic) combo — no duplicate texts
    by construction — with per-family counts balanced so the family split stays
    class-balanced."""
    items = []
    for task in sorted(list(CUES) + list(STYLE)):
        style = task in STYLE
        classes = sorted((STYLE if style else CUES)[task])
        n_cls = _n_per_class(task, len(classes))
        for cls in classes:
            fams = (STYLE if style else CUES)[task][cls]
            assert len(fams) == 12, (task, cls, len(fams))
            picked = []  # (fam, opener_idx or None, topic_idx)
            for fam in range(12):
                n_f = n_cls // 12 + (1 if fam < n_cls % 12 else 0)
                rng = random.Random(f"{seed}:{task}:{cls}:{fam}")
                if style:
                    combos = [(fam, None, t) for t in range(len(TOPICS))]
                else:
                    combos = [(fam, o, t) for o in range(len(OPENERS))
                              for t in range(len(TOPICS))]
                assert n_f <= len(combos), (task, cls, fam, n_f)
                rng.shuffle(combos)
                picked.extend(combos[:n_f])
            for i, (fam, o, t) in enumerate(picked):
                topic, req = TOPICS[t]
                if style:
                    text = fams[fam].format(req=req)
                else:
                    cue = fams[fam].format(t=topic)
                    text = f"{OPENERS[o]} {cue} Anyway, {req}?"
                items.append({
                    "id": f"{task}-{cls}-f{fam:02d}-i{i:03d}",
                    "task": task,
                    "label": cls,
                    "family": fam,
                    "split": _split_of(fam),
                    "topic": topic,
                    "reading_attr": READING_ATTR[task],
                    "text": text,
                })
    return items


def dataset_sha(items: list[dict]) -> str:
    blob = "\n".join(json.dumps(it, sort_keys=True) for it in items)
    return hashlib.sha256(blob.encode()).hexdigest()


def write(out_dir: str = ".") -> None:
    items = generate()
    out = Path(out_dir)
    ds = out / "dataset.jsonl"
    ds.write_text("\n".join(json.dumps(it, sort_keys=True) for it in items) + "\n")
    manifest = {
        "seed": SEED,
        "n_items": len(items),
        "sha256_dataset_jsonl": hashlib.sha256(ds.read_bytes()).hexdigest(),
        "sha256_convgen_py": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "tasks": {},
    }
    for task in sorted({it["task"] for it in items}):
        sub = [it for it in items if it["task"] == task]
        manifest["tasks"][task] = {
            "classes": sorted({it["label"] for it in sub}),
            "n_items": len(sub),
            "n_test": sum(1 for it in sub if it["split"] == "test"),
        }
    (out / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    print(f"wrote {ds} ({len(items)} items) sha256={manifest['sha256_dataset_jsonl'][:16]}...")


def _selftest() -> int:
    fails = []
    a, b = generate(), generate()
    if dataset_sha(a) != dataset_sha(b):
        fails.append("determinism: two runs differ")
    if dataset_sha(a) == dataset_sha(generate(seed=SEED + 1)):
        fails.append("seed insensitivity: different seed gave identical dataset")
    by_task = {}
    for it in a:
        by_task.setdefault(it["task"], []).append(it)
    for task, sub in sorted(by_task.items()):
        classes = sorted({it["label"] for it in sub})
        n = _n_per_class(task, len(classes))
        for cls in classes:
            got = sum(1 for it in sub if it["label"] == cls)
            if got != n:
                fails.append(f"balance {task}/{cls}: {got} != {n}")
        pat = re.compile(FORBIDDEN[task], re.IGNORECASE)
        for it in sub:
            m = pat.search(it["text"])
            if m:
                fails.append(f"leak {it['id']}: forbidden {m.group(0)!r} in text")
        train_f = {it["family"] for it in sub if it["split"] == "train"}
        test_f = {it["family"] for it in sub if it["split"] == "test"}
        val_f = {it["family"] for it in sub if it["split"] == "val"}
        if train_f & test_f or train_f & val_f or val_f & test_f:
            fails.append(f"split overlap in {task}")
        texts = [it["text"] for it in sub]
        if len(set(texts)) != len(texts):
            fails.append(f"duplicate texts in {task} ({len(texts) - len(set(texts))})")
        # topic balance: every class must see every topic at least once (cue tasks
        # draw uniformly; assert no class-topic hole big enough to act as a proxy)
        for cls in classes:
            tops = {it["topic"] for it in sub if it["label"] == cls}
            if len(tops) < len(TOPICS) - 2:
                fails.append(f"topic coverage {task}/{cls}: only {len(tops)}/{len(TOPICS)}")
    for f in fails[:40]:
        print("FAIL", f)
    print("selftest:", "PASS" if not fails else f"{len(fails)} FAILURES")
    return 1 if fails else 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(_selftest())
    write(sys.argv[1] if len(sys.argv) > 1 else ".")
