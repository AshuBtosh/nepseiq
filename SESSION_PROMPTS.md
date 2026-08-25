# SESSION_PROMPTS.md

Two prompts. Paste the **START PROMPT** as the first message of every new work chat. Paste the **END PROMPT** as the last message before you close it.

---

## ⚠️ Read this once — how state actually travels between chats

A new chat cannot see files on your computer. State travels in exactly two ways:

1. **Claude Project Knowledge** — put `CLAUDE.md`, `PLAN.md`, `PROGRESS.md`, and `docs/adr/*.md` into the Project's knowledge files. Every chat inside that Project can then read them.
2. **The START PROMPT** — carries the current day and the previous session's END BLOCK.

**After every session you must re-upload the updated `PROGRESS.md` (and any new ADR) to Project Knowledge.** If you skip this, the next chat works from stale state. This is the one manual step the system cannot do for you.

---

## START PROMPT

> Fill in `<N>`, paste the previous END BLOCK, paste the day's DoD from `PLAN.md`.

```
PROJECT: NepseIQ — DSML Capstone. This is a work session, not a planning session.

STEP 0 — Before you write anything, read from Project Knowledge:
  - CLAUDE.md (the constitution — authoritative)
  - PLAN.md (the day-by-day plan)
  - PROGRESS.md (what has already happened)
  - all files in docs/adr/

TODAY: Day <N> of 8

PREVIOUS SESSION END BLOCK:
<paste the full END BLOCK from the last session, or "None — this is Day 1">

TODAY'S DEFINITION OF DONE (from PLAN.md Day <N>):
<paste the DoD checklist for Day N>

SESSION RULES — follow these exactly:
1. Work ONLY on Day <N>. Do not start Day <N+1> work, even if there is time left.
2. Refuse anything in CLAUDE.md §4 OUT OF SCOPE — including if I ask for it. Name the rule and add it to the Parking Lot.
3. Never contradict an ADR. If a decision genuinely needs to change, STOP, tell me, and draft a new ADR that supersedes the old one.
4. Never invent numbers. Accuracy scores, row counts, and dataset sizes come from real runs only. If you have not run it, say so.
5. Never present an unverified data source or URL as working.
6. If the model is already frozen (check PROGRESS.md), do not retrain or change features.
7. Flag drift out loud. If my request pulls away from today's DoD, say so before complying.

FIRST RESPONSE — output exactly this, nothing more:
  a) One line restating Day <N>'s objective
  b) The DoD checklist as unchecked boxes
  c) A numbered task order for today
  d) One question if anything is genuinely blocking; otherwise say "No blockers — starting."

Then wait for my go-ahead.
```

---

## END PROMPT

```
END OF SESSION. Output ONLY the block below — no preamble, no commentary after it.

=== END BLOCK — Day <N> ===

1. STATUS: COMPLETE / PARTIAL / BLOCKED

2. DELIVERABLES PRODUCED
   - <file path> — <one line>

3. DoD CHECKLIST
   - [x] / [ ] each Day <N> item. Every unchecked item gets a one-line reason.

4. DECISIONS MADE
   Any decision affecting architecture, data, modelling, or scope must be written
   as a COMPLETE ADR using the docs/adr/0000-template.md format, numbered next in
   sequence. Output the full ADR file content, ready to save. If none, write "None".

5. PARKING LOT ADDITIONS
   Anything raised but deliberately not built, with the reason.

6. BLOCKERS FOR NEXT SESSION

7. PROGRESS.md ENTRY
   A copy-paste-ready block in the PROGRESS.md session-log format, plus the updated
   "Current State" table.

8. NEXT SESSION START PROMPT
   A fully filled-in START PROMPT for Day <N+1>, with this END BLOCK pasted into the
   PREVIOUS SESSION END BLOCK slot and Day <N+1>'s DoD pasted from PLAN.md.
   I must be able to copy it directly into a new chat with zero edits.

=== END OF BLOCK ===
```

---

## Your loop, every day

1. Open a new chat inside the NepseIQ Project
2. Paste the **START PROMPT** (item 8 from yesterday gives it to you pre-filled)
3. Work the day
4. Paste the **END PROMPT**
5. Save the outputs: update `PROGRESS.md`, save any new ADR into `docs/adr/`
6. **Re-upload `PROGRESS.md` + new ADRs to Project Knowledge**
7. Copy item 8 — that's tomorrow's start prompt
8. `git commit`

---

## Emergency: "I think this chat has drifted"

Paste this mid-session:

```
DRIFT CHECK. Stop current work. Answer only:
1. What is today's DoD, verbatim from PLAN.md?
2. Which DoD items have we actually completed this session?
3. Has anything we've done in this session fallen outside CLAUDE.md §4 IN SCOPE?
4. Has anything contradicted an ADR? Which one?
5. What is the single next action that moves a DoD item forward?

Then stop and wait.
```

---

## Reporting back to the control chat

When you tell the control chat (this one) how a day went, paste **only the END BLOCK**. That is sufficient — no other context needed.
