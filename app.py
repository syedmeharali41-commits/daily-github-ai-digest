import os
import requests
import smtplib
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

def get_github_repos():
    yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    url = f"https://api.github.com/search/repositories?q=created:>{yesterday}&sort=stars&order=desc&per_page=50"
    headers = {"Accept": "application/vnd.github.v3+json"}
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.json().get('items', [])
    return []

def analyze_with_ai(repos_data):
    groq_api_key = os.environ.get("GROQ_API_KEY")
    if not groq_api_key: return None
        
    simplified_repos = [{
        "name": r['full_name'],
        "description": r['description'] or "No description.",
        "stars": r['stargazers_count'],
        "language": r['language'] or "Unknown",
        "url": r['html_url']
    } for r in repos_data]

    prompt = f"""
    Analyze these GitHub repositories created in the last 24 hours. 
    Select the absolute best tool/repository for each of the following 6 categories:
    1. 🔥 The Trendsetter (Most Viral Repo)
    2. 💎 The Hidden Gem (Underrated but highly useful tool)
    3. 🛠️ Dev Utility & Automation (Tools that boost developer productivity)
    4. 🤖 AI & Data Science (Latest LLM wrappers, AI tools, or agents)
    5. 🎨 Frontend & UI/UX (Beautiful component libraries, CSS, or web designs)
    6. 🔒 CyberSecurity & DevOps (Deployment, privacy, or infrastructure tools)

    For each category, provide:
    - The Category Name with Emoji
    - Name of the repo hyperlinked to its GitHub URL
    - Current Star Count
    - A short, punchy 2-line description explaining WHAT it does and WHY it is useful.

    Return ONLY raw HTML inside a <div> structure. Do not include markdown code fences (```html). 
    Use clean inline CSS for a premium dark-mode dashboard look (background: #161b22, glowing accent text like #58a6ff or #7ed321, and modern card spacing).

    Data: {simplified_repos}
    """

    url = "[https://api.groq.com/openai/v1/chat/completions](https://api.groq.com/openai/v1/chat/completions)"
    headers = {"Authorization": f"Bearer {groq_api_key}", "Content-Type": "application/json"}
    data = {
        "model": "llama-3.3-70b-versatile",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2
    }
    
    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        return response.json()['choices'][0]['message']['content']
    return None

def send_email(html_content):
    sender = os.environ.get("EMAIL_SENDER")
    receiver = os.environ.get("EMAIL_RECEIVER")
    password = os.environ.get("EMAIL_PASSWORD")
    
    if not all([sender, receiver, password]): return

    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"✨ AI-Curated GitHub Repos - {datetime.now().strftime('%d %b, %Y')}"
    msg["From"] = sender
    msg["To"] = receiver

    full_html = f"""
    <html>
    <body style="background-color: #0d1117; color: #c9d1d9; font-family: system-ui, sans-serif; padding: 20px;">
        <div style="max-width: 600px; margin: 0 auto; background: #0d1117; padding: 10px;">
            <h1 style="color: #58a6ff; font-size: 24px; border-bottom: 1px solid #30363d; padding-bottom: 12px; margin-top: 0; font-weight: 600;">🤖 AI Curated Repo Digest</h1>
            <p style="color: #8b949e; font-size: 14px; margin-bottom: 25px;">Here is your morning intelligence report on the latest, most promising GitHub repositories.</p>
            {html_content}
            <div style="text-align: center; margin-top: 40px; border-top: 1px solid #30363d; padding-top: 20px; font-size: 11px; color: #8b949e; letter-spacing: 1px;">
                AUTOMATED VIA GITHUB ACTIONS & GROQ AI
            </div>
        </div>
    </body>
    </html>
    """
    msg.attach(MIMEText(full_html, "html"))
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender, password)
            server.sendmail(sender, receiver, msg.as_string())
        print("Email sent successfully!")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    raw_repos = get_github_repos()
    if raw_repos:
        ai_html = analyze_with_ai(raw_repos)
        if ai_html:
            send_email(ai_html)
