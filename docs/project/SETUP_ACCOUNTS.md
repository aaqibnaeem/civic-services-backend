# Accounts & keys you need to create

Do these now, in parallel with the code being built. Each takes 2–5 minutes. **Nothing here blocks
development** — the app runs locally on SQLite with the offline AI tiers until these exist.

Paste each value into `civic-backend/.env` (never commit that file — it is gitignored).

---

## 1. DeepSeek API key — powers the primary AI tier

1. Go to <https://platform.deepseek.com/> and sign up (phone verification, no card required).
2. Open **API Keys** → **Create new API key**. Copy it immediately; it is shown once.
3. Check **Top up / Balance**. New accounts usually get a free token grant. If your balance is zero, add the
   minimum top-up — **this entire project costs well under $2** at current `deepseek-v4-flash` pricing
   ($0.14 per million input tokens, $0.28 per million output).
4. Put it in `civic-backend/.env`:
   ```
   DEEPSEEK_API_KEY=sk-...
   ```

> If the balance is empty the API returns HTTP 402 and the app automatically falls back to the trained
> scikit-learn model, then to the rule engine. The demo still works — it just shows an `ML` badge instead of
> an `LLM` badge on the AI panel.

---

## 2. Neon Postgres — the production database

1. Go to <https://neon.tech/> → sign in with GitHub → **Create project**.
2. Name it `civic-services`, pick the region closest to you, accept the defaults.
3. Copy the **connection string** from the dashboard. It looks like:
   ```
   postgresql://user:password@ep-xxx.region.aws.neon.tech/neondb?sslmode=require&channel_binding=require
   ```
4. Paste it into `.env` as `DATABASE_URL`. **Paste it exactly as Neon gives it to you** — the backend
   normalises the driver prefix and strips the `sslmode`/`channel_binding` parameters that the async driver
   rejects. You do not need to edit the string by hand.

Free tier is permanent and needs no card. Locally you can ignore this entirely and the app uses a SQLite file.

---

## 3. GitHub — two repositories

The hackathon requires a public repo. We need **two**, because the frontend and backend are separate projects.

1. At <https://github.com/new> create `civic-services-backend` (public, **no** README/gitignore/licence —
   the folders already have them).
2. Repeat for `civic-services-frontend`.
3. Tell me the two URLs and I will wire up the remotes and push. (The `gh` CLI is not installed on this
   machine, so the repos have to be created in the browser.)

---

## 4. Render — backend hosting

1. Sign up at <https://render.com/> with GitHub and authorise access to the backend repo.
2. **New → Web Service** → pick `civic-services-backend`. The included `render.yaml` supplies the build and
   start commands, health check path and plan.
3. In **Environment**, add: `DATABASE_URL`, `DEEPSEEK_API_KEY`, `SECRET_KEY` (any long random string),
   `CORS_ORIGINS` (your Vercel URL, added after step 5), `SEED_ON_STARTUP=true`.
4. Deploy, then open `https://<your-service>.onrender.com/health`.

> **Free-tier warning that matters for your demo:** the instance sleeps after ~15 minutes idle and takes
> 30–60 seconds to wake. Open the URL a few minutes *before* you present so the judges never see a cold start.

---

## 5. Vercel — frontend hosting

1. Sign up at <https://vercel.com/> with GitHub → **Add New → Project** → import `civic-services-frontend`.
2. Framework preset **Vite**. Build `npm run build`, output `dist` (auto-detected).
3. Add the environment variable `VITE_API_URL = https://<your-service>.onrender.com/api/v1`.
4. Deploy, then copy the resulting URL back into Render's `CORS_ORIGINS` and redeploy the backend.

---

## Checklist

- [ ] `DEEPSEEK_API_KEY` in `.env`
- [ ] `DATABASE_URL` from Neon in `.env`
- [ ] Two empty GitHub repos created, URLs shared
- [ ] Render service live, `/health` returns OK
- [ ] Vercel site live, `VITE_API_URL` set, `CORS_ORIGINS` updated on Render
