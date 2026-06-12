import os
import sys
import json
import time
import logging
import requests
import smtplib
from datetime import datetime, timedelta, timezone
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# ----------------------------------------------------------------------------
# Logging setup
# ----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("github-ai-digest")

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
MIN_STARS = 5            # ignore brand-new repos with almost no traction
MAX_REPOS_TO_SEND_AI = 60 # cap payload size sent to the LLM
REQUEST_TIMEOUT = 30

GROQ_MODELS = [
    "llama-3.3-70b-versatile",   # primary
    "llama-3.1-8b-instant",      # fast fallback
]

CATEGORIES = [
    ("🔥 The Trendsetter", "Most viral / fastest growing repo overall"),
    ("💎 The Hidden Gem", "Underrated but highly useful, lower star count"),
    ("🛠️ Dev Utility & Automation", "Tools that boost developer productivity"),
    ("🤖 AI & Data Science", "LLM wrappers, AI agents, ML/data tools"),
    ("🎨 Frontend & UI/UX", "Component libraries, CSS, design systems"),
    ("🔒 CyberSecurity & DevOps", "Deployment, privacy, infra, security tools"),
]


# ----------------------------------------------------------------------------
# Step 1: Fetch trending repos from GitHub
# ----------------------------------------------------------------------------
def get_github_repos():
    """Fetch repos created in the last 24h, sorted by stars, with basic quality filtering."""
    yesterday = (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d")
    url = (
        "https://api.github.com/search/repositories"
        f"?q=created:>{yesterday}&sort=stars&order=desc&per_page=100"
    )
    headers = {"Accept": "application/vnd.github.v3+json"}

    github_token = os.environ.get("GITHUB_TOKEN")
    if github_token:
        headers["Authorization"] = f"Bearer {github_token}"

    for attempt in range(1, 4):
        try:
            resp = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
            if resp.status_code == 200:
                items = resp.json().get("items", [])
                log.info(f"Fetched {len(items)} repos from GitHub (attempt {attempt}).")
                return filter_repos(items)
            elif resp.status_code == 403:
                log.warning("GitHub rate limit hit. Retrying after delay...")
                time.sleep(5 * attempt)
            else:
                log.error(f"GitHub API error {resp.status_code}: {resp.text[:200]}")
                time.sleep(2 * attempt)
        except requests.RequestException as e:
            log.error(f"GitHub request failed (attempt {attempt}): {e}")
            time.sleep(2 * attempt)

    log.error("Failed to fetch repos from GitHub after retries.")
    return []


def filter_repos(items):
    """Filter out forks, archived, and very low-quality repos. Quality > raw count."""
    filtered = []
    for r in items:
        if r.get("fork"):
            continue
        if r.get("archived"):
            continue
        if r.get("stargazers_count", 0) < MIN_STARS:
            continue
        if not r.get("description"):
            continue
        filtered.append(r)

    filtered.sort(key=lambda r: r.get("stargazers_count", 0), reverse=True)
    log.info(f"{len(filtered)} repos passed quality filter (min_stars={MIN_STARS}, no forks/archived).")
    return filtered[:MAX_REPOS_TO_SEND_AI]


# ----------------------------------------------------------------------------
# Step 2: Analyze with AI (Groq, with model fallback)
# ----------------------------------------------------------------------------
def build_prompt(simplified_repos):
    category_lines = "\n".join(
        f"{i+1}. {name} — {desc}" for i, (name, desc) in enumerate(CATEGORIES)
    )

    return f"""
You are curating a daily "Best of GitHub" digest for a developer audience.

Analyze these GitHub repositories created in the last 24 hours.
Select the single best repository for EACH of the following categories.
If no repo genuinely fits a category well, choose the closest reasonable match —
never leave a category empty, but do not force a bad fit if a better
alternative exists in another category's pool.

Categories:
{category_lines}

For each category, output ONE <tr> table row with exactly this structure:

<tr style="border-bottom:1px solid #e0e0e0;">
  <td style="padding:16px 12px; vertical-align:top;">
    <div style="font-weight:600; font-size:13px; color:#888888; margin-bottom:6px; text-transform:uppercase; letter-spacing:0.5px;">CATEGORY EMOJI + NAME HERE</div>
    <a href="REPO_URL" target="_blank" style="font-size:15px; font-weight:600; color:#1a6496; text-decoration:none;">REPO FULL NAME</a>
    <p style="margin:8px 0 10px 0; font-size:13px; color:#444444; line-height:1.6;">
      Write 2 punchy sentences: WHAT it does and WHY a developer would care. Use your own words, not copy-paste.
    </p>
    <span style="display:inline-block; background:#f0f0f0; border:1px solid #d0d0d0; border-radius:4px; padding:2px 8px; font-size:11px; color:#555555; margin-right:6px;">LANGUAGE</span>
    <span style="display:inline-block; background:#fff3cd; border:1px solid #ffc107; border-radius:4px; padding:2px 8px; font-size:11px; color:#856404;">⭐ STAR_COUNT stars</span>
  </td>
  <td style="padding:16px 12px; vertical-align:middle; text-align:center; width:130px; white-space:nowrap;">
    <a href="REPO_URL" target="_blank" style="display:inline-block; background:#1db954; color:#ffffff; font-weight:700; font-size:13px; text-decoration:none; padding:9px 18px; border-radius:5px;">Go to Repo →</a>
  </td>
</tr>

Rules:
- Return ONLY the raw <tr> rows — no markdown, no code fences, no wrapping tags
- No <table>, no <html>, no <body> — just the <tr> elements back to back
- Fill in all placeholder values (CATEGORY, REPO URL, REPO FULL NAME, LANGUAGE, STAR COUNT) from the data
- The description must be YOUR OWN words — informative and punchy, not copied from the repo description

Repository data (JSON):
{json.dumps(simplified_repos, ensure_ascii=False)}
"""


def call_groq(prompt, api_key, model):
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    data = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "max_tokens": 4096,
    }

    resp = requests.post(url, headers=headers, json=data, timeout=60)
    if resp.status_code == 200:
        content = resp.json()["choices"][0]["message"]["content"]
        return clean_html_response(content)

    log.warning(f"Groq model '{model}' failed: {resp.status_code} {resp.text[:200]}")
    return None


def clean_html_response(content):
    """Strip markdown code fences / stray wrappers the model might add."""
    content = content.strip()
    if content.startswith("```"):
        content = content.split("\n", 1)[-1]  # drop first fence line
    if content.endswith("```"):
        content = content.rsplit("```", 1)[0]
    return content.strip()


def analyze_with_ai(repos_data):
    groq_api_key = os.environ.get("GROQ_API_KEY")
    if not groq_api_key:
        log.error("GROQ_API_KEY not set. Skipping AI analysis.")
        return None

    simplified_repos = [
        {
            "name": r["full_name"],
            "description": r.get("description") or "No description provided.",
            "stars": r.get("stargazers_count", 0),
            "language": r.get("language") or "Unknown",
            "url": r["html_url"],
        }
        for r in repos_data
    ]

    prompt = build_prompt(simplified_repos)

    for model in GROQ_MODELS:
        log.info(f"Requesting AI analysis using model: {model}")
        result = call_groq(prompt, groq_api_key, model)
        if result:
            log.info(f"AI analysis succeeded with model: {model}")
            return result
        time.sleep(2)

    log.error("All Groq models failed. No AI content generated.")
    return None


# ----------------------------------------------------------------------------
# Step 3: Build and send the email
# ----------------------------------------------------------------------------
def build_email_html(ai_rows, repo_count):
    today_str = datetime.now().strftime("%A, %d %B %Y")

    return f"""\
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
</head>
<body style="margin:0; padding:0; background-color:#f2f2f2; font-family:-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;">
  <table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="background-color:#f2f2f2; padding:24px 0;">
    <tr>
      <td align="center">
        <table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="max-width:680px; background-color:#ffffff; border:1px solid #d6d6d6; border-radius:8px; overflow:hidden;">

          <!-- Header -->
          <tr>
            <td style="padding:20px 24px; background-color:#1a1a2e; border-bottom:3px solid #1db954;">
              <h1 style="color:#ffffff; font-size:20px; font-weight:700; margin:0 0 4px 0;">
                🤖 AI-Curated GitHub Digest
              </h1>
              <p style="color:#aaaaaa; font-size:12px; margin:0;">
                {today_str} &nbsp;•&nbsp; {repo_count} fresh repos analyzed in the last 24 hours
              </p>
            </td>
          </tr>

          <!-- Table Header Row -->
          <tr>
            <td style="padding:0;">
              <table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="border-collapse:collapse;">
                <thead>
                  <tr style="background-color:#2d2d2d;">
                    <th style="padding:10px 12px; text-align:left; font-size:11px; font-weight:700; color:#cccccc; letter-spacing:1px; text-transform:uppercase;">PROJECT DESCRIPTION</th>
                    <th style="padding:10px 12px; text-align:center; font-size:11px; font-weight:700; color:#cccccc; letter-spacing:1px; text-transform:uppercase; width:130px;">LINK</th>
                  </tr>
                </thead>
                <tbody>
                  {ai_rows}
                </tbody>
              </table>
            </td>
          </tr>

          <!-- Footer -->
          <tr>
            <td style="padding:14px 24px; border-top:1px solid #e0e0e0; text-align:center; background-color:#f9f9f9;">
              <p style="color:#999999; font-size:11px; letter-spacing:1px; margin:0; text-transform:uppercase;">
                Automated via GitHub Actions &amp; Groq AI
              </p>
            </td>
          </tr>

        </table>
      </td>
    </tr>
  </table>
</body>
</html>
"""


def send_email(html_content, repo_count):
    sender = os.environ.get("EMAIL_SENDER")
    receiver = os.environ.get("EMAIL_RECEIVER")
    password = os.environ.get("EMAIL_PASSWORD")

    if not all([sender, receiver, password]):
        log.error("Missing email credentials (EMAIL_SENDER / EMAIL_RECEIVER / EMAIL_PASSWORD). Aborting send.")
        return False

    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"✨ AI-Curated GitHub Repos — {datetime.now().strftime('%d %b, %Y')}"
    msg["From"] = sender
    msg["To"] = receiver

    full_html = build_email_html(html_content, repo_count)

    # Plain-text fallback for clients that block HTML
    plain_text = (
        f"AI-Curated GitHub Digest — {datetime.now().strftime('%d %b, %Y')}\n\n"
        f"{repo_count} fresh repos analyzed.\n\n"
        "View this email in an HTML-capable client for the full curated list with links."
    )

    msg.attach(MIMEText(plain_text, "plain"))
    msg.attach(MIMEText(full_html, "html"))

    for attempt in range(1, 4):
        try:
            with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=REQUEST_TIMEOUT) as server:
                server.login(sender, password)
                server.sendmail(sender, receiver, msg.as_string())
            log.info("Email sent successfully!")
            return True
        except smtplib.SMTPAuthenticationError as e:
            log.error(f"SMTP authentication failed: {e}")
            return False  # retrying won't help with bad credentials
        except Exception as e:
            log.warning(f"Email send failed (attempt {attempt}): {e}")
            time.sleep(3 * attempt)

    log.error("Failed to send email after retries.")
    return False


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    log.info("Starting daily GitHub AI digest run...")

    raw_repos = get_github_repos()
    if not raw_repos:
        log.warning("No repos found matching criteria. Exiting without sending email.")
        return

    ai_html = analyze_with_ai(raw_repos)
    if not ai_html:
        log.error("AI analysis returned nothing. Exiting without sending email.")
        return

    success = send_email(ai_html, len(raw_repos))
    if not success:
        sys.exit(1)

    log.info("Run completed successfully.")


if __name__ == "__main__":
    main()
