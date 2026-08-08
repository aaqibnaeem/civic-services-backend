---
phase: 09
title: Deploy, Demo & Submit
status: not_started
owner: lead + user
depends_on: [08]
started:
completed:
---

# Phase 09 — Deploy, Demo & Submit

## Goal
Get both halves publicly reachable, then produce every artifact the hackathon asks for.

## Tasks
- [ ] Create the two GitHub repos, push `civic-backend` and `civic-frontend`
- [ ] Neon database created, `DATABASE_URL` set on Render
- [ ] Render web service live, `/health` green, seed data loaded
- [ ] Vercel site live, `VITE_API_URL` pointed at Render
- [ ] `CORS_ORIGINS` on Render updated with the Vercel origin, backend redeployed
- [ ] End-to-end smoke test against the **public** URLs, not localhost
- [ ] README links filled in (live URLs, repo URLs)
- [ ] UI screenshots captured: citizen submit, AI reveal, tracking, admin inbox, complaint detail, dashboard
- [ ] Demo video recorded (3–5 min)

## Acceptance criteria
- [ ] A complaint submitted from the Vercel URL appears in the admin inbox with an AI classification
- [ ] The analytics dashboard renders real numbers with written interpretations
- [ ] `/docs` on the Render URL loads the interactive API documentation

## Pre-demo warm-up (do this 5 minutes before presenting)
Render's free instance sleeps after ~15 minutes idle and takes 30–60s to wake. **Open the API URL and the
site, submit one throwaway complaint, and load the dashboard before you present.** A cold start in front of
judges reads as a broken app.

## Demo script — 5 minutes, hits every rubric line

| Time | Show | Rubric points earned |
|---|---|---|
| 0:00–0:30 | The problem: a real Karachi complaint in plain text, and why free text is useless to a service team | Problem understanding (10) |
| 0:30–1:30 | Submit a live complaint. Let the AI panel classify it on screen — category, priority, summary, department, confidence | AI implementation (25) |
| 1:30–2:00 | Show the stored complaint and its reference code; track it as a citizen with no account | Backend, UI/UX |
| 2:00–2:45 | Admin inbox: filter to critical + open, search, sort, open the detail, reassign, change status, watch the timeline update | Backend (15), UI/UX (10) |
| 2:45–3:15 | Duplicate detection catching a second report of the same problem | AI advanced option |
| 3:15–4:15 | Analytics dashboard. Do **not** read numbers aloud — read the interpretations. Point at the resolution-time box plot and explain why the median beats the mean here, and what the upper fence is flagging | Statistics (15) |
| 4:15–4:40 | Pull the API key or show `/ai/health`: the same complaint still gets classified by the trained model, badge changes from LLM to ML. Explain the three tiers | AI implementation, honesty |
| 4:40–5:00 | Architecture diagram and class model; state the AI's limitations plainly | OOP (10), Presentation (5) |

## Questions judges will ask — and the honest answers
- **"What does the AI actually receive and return?"** Raw complaint text plus location context in; a validated
  JSON object with category, priority, summary, department, confidence and keywords out.
- **"How accurate is it?"** Give the measured macro-F1 from the evaluation report, then immediately say the
  test set is synthetic and therefore an upper bound. Never claim perfect accuracy.
- **"What happens when the API is down?"** Show it. That is what the tier badge is for.
- **"Why is this a class and not a function?"** The three analyzers are runtime-substitutable behind one
  interface; the pipeline selects by availability without knowing which it got.
- **"Why median instead of mean?"** Because resolution time is right-skewed; the dashboard says so in words.

## Submission checklist
- [ ] GitHub repository with clear structure
- [ ] Public deployment URL
- [ ] README: problem, features, architecture, AI technology, setup, usage
- [ ] Architecture diagram → [ARCHITECTURE.md](../ARCHITECTURE.md)
- [ ] AI testing evidence: example inputs, outputs, limitations
- [ ] UI screenshots, citizen and admin
- [ ] 3–5 minute demo video
- [ ] **No API keys committed** — verify with a final `git log -p | grep -i "sk-"` before pushing
