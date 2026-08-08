# Demo & Presentation Guide

Everything you need to test the project, present it, and answer the judges. Written in plain language.
Every number below was read from the **live deployed system**, not made up.

| | |
|---|---|
| **App (citizens + staff)** | https://civic-services-frontend.vercel.app |
| **API** | https://civic-services-backend.onrender.com |
| **API documentation** | https://civic-services-backend.onrender.com/docs |
| **Staff login** | `admin@civic.gov.pk` / `Admin@123` |

---

## 1. Ten minutes before you present

Render's free server goes to sleep after about 15 minutes of no traffic, and takes 30–60 seconds to wake
up. If a judge sees that wait, it looks like the app is broken. So:

1. **Wake the API.** Open https://civic-services-backend.onrender.com/health in a tab.
   You want to see `"status": "ok"` and `"database": "ok"`. If it takes a minute the first time, that is
   the server waking up — that is exactly why you do this early.
2. **Open the app** and let the home page load. You should see real numbers (around 800 complaints), not
   error boxes.
3. **Log in as staff** in a second tab so you are not typing a password in front of people.
4. **Open the dashboard** once, so the charts are already loaded and cached.
5. Have this guide open on your phone or a second screen.

If the home page shows *"Live statistics are unavailable"*, the app cannot reach the API. Check that
`VITE_API_URL` is set in Vercel and that you redeployed after setting it.

---

## 2. The project in one sentence

> A citizen describes a local problem in their own words, and the system understands it, decides how urgent
> it is, sends it to the right department, and then measures how well the city is responding.

If you only get to say one thing, say that.

---

## 3. The demo — about six minutes

### Minute 0:00–0:30 — The problem

Say this in your own words:

> "In Karachi, people report broken streetlights, uncollected rubbish, potholes and burst water pipes.
> They write things like *'kachra ka dher gali ke corner par pade hue hain'*. That sentence is useless to a
> service team. Nobody knows which department it belongs to, or whether it is more urgent than the other
> three hundred complaints waiting. This project turns that sentence into something a team can act on."

### 0:30–2:00 — Submit a complaint and watch the AI work

Go to **Report an issue**. Type a real complaint. A good one for the demo:

> `Bijli ka khamba Sector 11-B mein jhuk gaya hai aur taarein neeche latak rahi hain, barish ka pani bhi
> jama hai wahan aur bachay khelte hain. Bohat khatarnak hai.`

*(A leaning electricity pole with hanging wires, standing rainwater underneath, children playing there.)*

Fill in the location, then move to the **review step**. The AI reads it live on screen.

**What to point at:**
- It picked the category **Electricity** — even though the complaint is written in Roman Urdu.
- It set the priority to **Critical**, and flagged it as an **emergency**.
- It wrote a one-line summary for the service team.
- It chose the responsible **department** on its own.
- There is a **badge showing which AI produced this** — this matters, and you will come back to it.
- The **confidence score** shows how sure it is.

Say:

> "It did not just match keywords. It understood that hanging live wires above standing water, where
> children play, is a danger to life — and that is why it said critical."

Then point out the **category override** control:

> "The AI can be wrong, so the citizen can correct it. If they do, their choice wins — the system will not
> silently overrule a person. But the AI's own opinion is still saved, so staff can see the disagreement."

Submit it. **Copy the reference code** (it looks like `CIV-D4GNAW`).

### 2:00–2:30 — Track it as a citizen

Go to **Track**, paste the code.

> "The citizen never made an account. There is no signup, no password. The reference code is enough, and
> the browser remembers the codes you have filed so you can find them again."

### 2:30–3:30 — The staff side

Switch to the tab where you are already logged in as staff.

- Show the **inbox**: around 800 complaints.
- **Filter** to Critical + Open. Show the count drop.
- **Search** for a word like "transformer".
- **Open a complaint**. Show the AI panel, the confidence, and the timeline.
- **Change its status** to In Progress. Point out the timeline updates with who did it and when.
- Show the **possible duplicates** panel: *"this looks like the same problem someone else already reported."*

### 3:30–5:00 — The dashboard (this is worth the most marks after the AI)

Open **Analytics**. **Do not read the numbers out.** Read the *sentences*. The system writes them itself:

- *"20 critical complaints are still unresolved right now."*
- *"160 open complaints (64% of the backlog) have been waiting more than 14 days."*
- *"55% of complaints are rated high or critical — the priority scale has lost its meaning."*
- *"37 complaints took longer than 13.5 days — far beyond the normal range."*

Then show the **resolution time** section and say:

> "Half of all complaints are closed within 3.3 days. But the average says 5.4 days. The average is higher
> because a small number of very slow cases drag it up. So the honest number to report is the median, not
> the average — and the dashboard says that in words, not just in numbers."

Point at the box plot:

> "This line is the upper fence. Anything past 13.5 days is statistically abnormal — not just slow, but
> outside the normal range. There are 37 of them, and the system lists them so a manager can go and look at
> those specific cases."

### 5:00–5:30 — The safety net (this is your strongest moment)

Say:

> "Everything you just saw depends on an AI API. What happens if it goes down during a demo — or during a
> real emergency?"

Open https://civic-services-backend.onrender.com/api/v1/ai/health and show the three tiers.

> "There are three. First, DeepSeek. If that fails, a machine-learning model we trained ourselves takes
> over — it runs on our own server with no internet. If even that is missing, a rule-based engine handles
> it. It always answers. And every complaint records **which** of the three answered it, so we never
> pretend a keyword guess came from the AI."

### 5:30–6:00 — Close

Show the architecture diagram in the repo, then finish with the limitations (section 7 below). Saying your
own weaknesses out loud earns more marks than pretending there are none.

---

## 4. How the AI works — explain it this simply

The brief says you must be able to explain what the AI receives, what it does, what it returns, and where
it fails.

**What it receives:** the complaint text the citizen typed, plus the location they gave.

**What it does:** it sends that text to DeepSeek (`deepseek-v4-flash`) with instructions listing the seven
categories, the four urgency levels, and the departments — plus a few worked examples. It is asked to reply
in a strict format.

**What it returns:** category, priority, a one-line summary, the responsible department, a confidence
score, keywords, and whether it is an emergency. All of that is saved with the complaint.

**Why we don't just trust it:** the reply is checked against a strict schema before we accept it. If the AI
returns something malformed, we ask once more, then fall back to our own model. The AI is treated as
untrusted input, like anything else coming off the internet.

**The three tiers, in order:**

| Tier | What it is | When it runs | How fast |
|---|---|---|---|
| 1 | DeepSeek `deepseek-v4-flash` | Normally | ~2 seconds |
| 2 | Our own trained model (TF-IDF + LinearSVC) | If DeepSeek fails or has no key | 8 milliseconds |
| 3 | Keyword and rule engine | If the model file is missing | under 1 millisecond |

**One more safety rule:** the keyword engine is allowed to *raise* an urgency level but never lower it.
This exists because during testing the trained model rated a leaning pole with live wires over water as
"low". Now a hazard word can escalate that, but nothing can quietly downgrade a danger.

---

## 5. What the statistics actually say

Live numbers from the deployed system, over 537 resolved complaints:

| Measure | Value | What it means in plain words |
|---|---|---|
| Median | 78.3 hours (3.3 days) | Half of all complaints are done faster than this |
| Mean (average) | 128.4 hours (5.4 days) | Pulled upwards by a few very slow cases |
| Mode | 10.3 hours | The single most common resolution time |
| Minimum / Maximum | 3.4 h / 1520.3 h | Fastest and slowest ever |
| Range | 1516.9 hours | The gap between them |
| Variance | 25,992.5 | How spread out the values are |
| Standard deviation | 161.2 hours | Typical distance from the average |
| Q1 / Q3 | 41.5 h / 154.3 h | The middle half sits between these |
| IQR | 112.8 hours | The width of that middle half |
| Upper fence | 323.4 h (13.5 days) | Past this, a case is an outlier |
| Skewness | 4.19 | Strongly lopsided towards slow cases |

**The one insight to explain:** the mean is 64% higher than the median. That is what skewness of 4.19 means
in practice — the data has a long tail of slow cases. So the median is the honest headline number. A lot of
teams will just print "average resolution time" and be wrong. You can explain why you didn't.

**The statistical test:** we checked whether the category of a complaint is related to how urgent it is.

> Chi-square test of independence: χ²(18) = 160.3, p < 0.001, Cramér's V = 0.26

In plain words: **yes, they are related, and it is a medium-strength relationship.** Some kinds of problems
really are more urgent than others. We also checked the test was valid to run — the smallest expected count
was 5.7, and none were below 5, which is the condition the test requires. The dashboard says so.

**Honesty built in:** the system also warns you that resolution times are measured only on complaints that
were actually resolved. The slow ones are the ones most likely still open — so every speed number here is
slightly optimistic. It says that on screen.

---

## 6. How this covers what the brief asked for

| The brief asked for | Where it is | How to show it |
|---|---|---|
| Citizen submits complaint | Report page | Submit one live |
| AI analyses it | Review step of the form | Watch it classify on screen |
| Predict category | 7 categories | The badge on the result |
| Predict priority | Low/Medium/High/Critical | The priority badge |
| Summarise for the team | One-line summary | Shown in the AI panel |
| Store everything | Postgres on Neon | Open the complaint in the inbox |
| Admin can view, filter, assign, update | Staff inbox + detail page | Filter, then change a status |
| Dashboard with statistics and trends | Analytics page | Insights, charts, box plot |
| Search and filters | Inbox | Search "transformer", filter by priority |
| Error handling | Everywhere | Try a 3-letter complaint; try an invalid code |
| Object-oriented design | `app/services/`, `app/ai/` | The class diagram in ARCHITECTURE.md |
| Deployed publicly | Vercel + Render | You are demoing on it |
| Data preparation and evaluation | `ml/` folder | `evaluation.md` with real scores |
| Duplicate detection *(bonus)* | Complaint detail page | The duplicates panel |
| Department routing *(bonus)* | Automatic on submit | The department was chosen for you |
| AI assistant *(bonus)* | Assistant page | Ask "which area has the most drainage complaints?" |

**Three benchmarks in one project:**
- *Advanced AI* — a real trained model with honest evaluation, plus a live LLM, plus duplicate detection.
- *Statistics* — every measure the brief listed, with quartiles, fences, a significance test, and written
  explanations.
- *Object-oriented* — `AIAnalyzer` is an abstract base class with three real subclasses that swap at
  runtime; `ComplaintManager` owns the rules about status changes.

---

## 7. Questions judges will ask

**"How accurate is it?"**
> On 40 complaints we wrote by hand, DeepSeek got the category right every time and the priority right 78%
> of the time. Our own trained model gets 75.8% on category. But our training data is generated, not real,
> so that number is a best case — real complaints are messier. That is exactly why the trained model is the
> backup and not the main one.

**"Isn't this just a wrapper around an API?"**
> No. We trained our own model and can show its confusion matrix. We built duplicate detection, department
> routing, and a statistics engine. And the AI is one of three layers, not the whole system.

**"What if the AI is down?"**
> Show `/ai/health` and explain the three tiers. Also: submitting a complaint never waits for the AI. It
> saves instantly and the AI fills in afterwards, so an outage can delay analysis but can never lose a
> citizen's report.

**"Could the AI make up statistics?"**
> No. Every sentence on the dashboard is generated by fixed rules from computed numbers, not written by the
> AI. The AI is never asked to count anything. In the assistant, the AI decides *what* to look up, our code
> does the counting, and then the AI only writes the sentence around numbers we calculated.

**"Why is that a class and not just a function?"**
> Because the three analysers are interchangeable. The pipeline asks for whichever one is available and
> does not need to know which it got. That is what an abstract base class is for.

**"Why median instead of average?"**
> Because resolution time is heavily skewed — skewness 4.19. The average is 64% higher than the median
> because of a few very slow cases. The median is the honest number.

**"Is your API key safe?"**
> Yes. It only exists in the server's environment variables. The browser never sees it, and it was never
> committed to GitHub.

---

## 8. Limitations — say these before they ask

This is worth marks. Overclaiming loses them.

1. **The training data is synthetic.** We generated it. So the model's test score is an upper bound, not a
   promise about real-world accuracy.
2. **Priority is subjective.** Two reasonable people would disagree on whether something is High or
   Critical. Our accuracy figure reflects agreement with our own labels, not objective truth.
3. **Duplicate detection compares words, not meaning.** Two people describing the same pothole in
   completely different words will not be matched.
4. **No image understanding.** DeepSeek's API does not accept images, so photo analysis is not built.
5. **The AI is not deterministic.** The same complaint can occasionally get a slightly different result.
6. **Speed statistics are optimistic**, because only resolved complaints have a resolution time and slow
   cases are the ones most likely still open.
7. **The free server sleeps**, so the first request after a quiet period is slow.

---

## 9. Test it yourself before demo day

Tick these off:

- [ ] `/health` returns `"status": "ok"` and `"database": "ok"`
- [ ] Home page shows around 800 complaints, not an error box
- [ ] Submit a complaint → AI result appears → you get a reference code
- [ ] Track that code → it shows the complaint and its AI analysis
- [ ] Staff login works
- [ ] Inbox filters change the result count
- [ ] Changing a status adds a line to the timeline
- [ ] Dashboard loads with charts and written insights
- [ ] Assistant answers a question and cites complaint codes
- [ ] Try a 3-character complaint → you get a clear error, not a crash
- [ ] Try reference code `CIV-XXXXXX` → clear "not found", not a crash
- [ ] Open the site on your phone → still usable
