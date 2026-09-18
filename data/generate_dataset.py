"""
Dataset generator: creates a synthetic but realistic phishing/legitimate URL dataset.
Run this ONCE to produce data/phishing_dataset.csv if you don't have a real dataset.

Real dataset option (recommended):
  Download from: https://www.kaggle.com/datasets/shashwatwork/phishing-dataset-for-machine-learning
  Place the CSV as: data/phishing_dataset.csv
  Columns needed: url (string), label (0=legitimate, 1=phishing)
"""

import csv
import random
import os

random.seed(42)

LEGIT_DOMAINS = [
    "google.com", "youtube.com", "facebook.com", "amazon.com", "wikipedia.org",
    "twitter.com", "instagram.com", "linkedin.com", "reddit.com", "github.com",
    "microsoft.com", "apple.com", "netflix.com", "spotify.com", "stackoverflow.com",
    "medium.com", "nytimes.com", "bbc.com", "cnn.com", "reuters.com",
    "yahoo.com", "bing.com", "dropbox.com", "salesforce.com", "adobe.com",
    "paypal.com", "ebay.com", "etsy.com", "shopify.com", "wordpress.com",
]

LEGIT_PATHS = [
    "/", "/about", "/contact", "/products", "/services", "/blog",
    "/news", "/help", "/support", "/pricing", "/terms", "/privacy",
    "/search?q=python", "/user/profile", "/dashboard", "/settings",
]

PHISHING_TRICKS = [
    "paypal-secure-login.{tld}",
    "amazon-account-verify.{tld}",
    "{brand}-billing-update.{tld}",
    "secure-{brand}-login.{tld}",
    "{brand}.com.{tld}",
    "account-{brand}-verify.{tld}",
    "login-{brand}-secure.{tld}",
    "{brand}-helpdesk-support.{tld}",
]

PHISHING_TLDS = ["xyz", "tk", "ml", "ga", "cf", "gq", "top", "click", "link", "info"]
BRANDS = ["paypal", "amazon", "netflix", "apple", "microsoft", "ebay", "bank", "chase"]
PHISHING_PATHS = [
    "/login/verify?cmd=account&update=1",
    "/secure/billing/confirm",
    "/account/suspended?redirect=http://evil.com",
    "/signin/password/reset",
    "/webscr?cmd=_login-submit",
    "/update/account/billing",
    "/confirm/identity/verify",
    "/download/install/setup.exe",
]
PHISHING_SUBDOMAINS = [
    "secure", "login", "account", "verify", "update", "mail", "banking",
    "support", "helpdesk", "service", "validation", "confirm",
]
IP_RANGES = [
    "192.168.{}.{}", "10.{}.{}.{}", "172.{}.{}.{}",
    "185.{}.{}.{}", "45.{}.{}.{}", "194.{}.{}.{}",
]


def make_legit_url():
    domain = random.choice(LEGIT_DOMAINS)
    scheme = "https" if random.random() > 0.1 else "http"
    path = random.choice(LEGIT_PATHS)
    sub = random.choice(["www.", "m.", ""]) if random.random() > 0.3 else ""
    return f"{scheme}://{sub}{domain}{path}", 0


def make_phishing_url():
    style = random.randint(0, 5)

    if style == 0:
        # IP-based phishing
        ip_template = random.choice(IP_RANGES)
        parts = [random.randint(1, 254) for _ in range(ip_template.count("{}"))]
        ip = ip_template.format(*parts)
        path = random.choice(PHISHING_PATHS)
        return f"http://{ip}{path}", 1

    elif style == 1:
        # Subdomain trick
        brand = random.choice(BRANDS)
        sub = random.choice(PHISHING_SUBDOMAINS)
        tld = random.choice(PHISHING_TLDS)
        path = random.choice(PHISHING_PATHS)
        return f"http://{sub}.{brand}-secure.{tld}{path}", 1

    elif style == 2:
        # Brand impersonation in domain
        brand = random.choice(BRANDS)
        trick = random.choice(PHISHING_TRICKS)
        tld = random.choice(PHISHING_TLDS)
        domain = trick.format(brand=brand, tld=tld)
        path = random.choice(PHISHING_PATHS)
        return f"http://{domain}{path}", 1

    elif style == 3:
        # URL shortener
        shorteners = ["bit.ly", "tinyurl.com", "goo.gl", "is.gd", "ow.ly"]
        short = random.choice(shorteners)
        code = "".join(random.choices("abcdefghijklmnopqrstuvwxyz0123456789", k=7))
        return f"http://{short}/{code}", 1

    elif style == 4:
        # @ symbol redirect
        legit = random.choice(LEGIT_DOMAINS)
        evil_tld = random.choice(PHISHING_TLDS)
        path = random.choice(PHISHING_PATHS)
        return f"http://{legit}@evil-phish.{evil_tld}{path}", 1

    else:
        # Long suspicious URL
        brand = random.choice(BRANDS)
        keywords = "-".join(random.choices(
            ["secure", "login", "verify", "update", "account", "billing", "confirm"],
            k=random.randint(2, 4)
        ))
        tld = random.choice(PHISHING_TLDS)
        path = random.choice(PHISHING_PATHS)
        num = random.randint(1000, 9999)
        return f"http://{brand}-{keywords}-{num}.{tld}{path}", 1


def generate_dataset(n_legit=5000, n_phishing=5000, output_path="data/phishing_dataset.csv"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    rows = []

    for _ in range(n_legit):
        url, label = make_legit_url()
        rows.append({"url": url, "label": label})

    for _ in range(n_phishing):
        url, label = make_phishing_url()
        rows.append({"url": url, "label": label})

    random.shuffle(rows)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["url", "label"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Dataset generated: {output_path} ({len(rows)} rows)")
    return output_path


if __name__ == "__main__":
    generate_dataset()
