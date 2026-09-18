"""
URL Feature Extraction for Phishing Detection
Extracts structural/lexical features from a URL string without visiting the page.
"""

import re
import urllib.parse
import ipaddress
import tldextract


# ── Suspicious keyword list ──────────────────────────────────────────────────
SUSPICIOUS_KEYWORDS = [
    "login", "verify", "update", "secure", "account", "bank", "paypal",
    "ebay", "amazon", "signin", "password", "confirm", "billing", "support",
    "helpdesk", "service", "customer", "validation", "submit", "free",
    "lucky", "winner", "prize", "click", "here", "download", "install",
    "urgent", "alert", "warning", "suspended", "blocked", "limited",
    "webscr", "cmd", "dispatch", "redirect"
]

# ── Known URL shorteners ─────────────────────────────────────────────────────
URL_SHORTENERS = [
    "bit.ly", "tinyurl.com", "goo.gl", "ow.ly", "t.co", "is.gd",
    "buff.ly", "adf.ly", "shorte.st", "bc.vc", "mcaf.ee", "tiny.cc",
    "rb.gy", "cutt.ly", "shorturl.at", "snip.ly"
]


def _is_ip_address(host: str) -> int:
    """Return 1 if host is a raw IPv4/IPv6 address, else 0."""
    try:
        ipaddress.ip_address(host)
        return 1
    except ValueError:
        return 0


def _count_subdomains(extracted) -> int:
    """Count number of subdomains (dot-separated parts before registered domain)."""
    sub = extracted.subdomain
    if not sub:
        return 0
    return len(sub.split("."))


def _uses_non_standard_port(parsed) -> int:
    """Return 1 if a non-standard port is explicitly specified."""
    port = parsed.port
    if port is None:
        return 0
    return 0 if port in (80, 443) else 1


def _has_redirection(url: str) -> int:
    """Return 1 if '//' appears in the URL path (excluding the scheme part)."""
    # Strip scheme, then look for '//'
    stripped = re.sub(r"^https?://", "", url, flags=re.IGNORECASE)
    return 1 if "//" in stripped else 0


def _digit_to_letter_ratio(url: str) -> float:
    """Ratio of digit characters to letter characters in the URL."""
    digits = sum(c.isdigit() for c in url)
    letters = sum(c.isalpha() for c in url)
    if letters == 0:
        return 1.0
    return round(digits / letters, 4)


def _suspicious_keyword_count(url: str) -> int:
    """Count how many suspicious keywords appear in the URL (case-insensitive)."""
    lower = url.lower()
    return sum(1 for kw in SUSPICIOUS_KEYWORDS if kw in lower)


def _is_url_shortener(extracted) -> int:
    """Return 1 if the registered domain belongs to a known URL shortener."""
    domain = f"{extracted.domain}.{extracted.suffix}".lower()
    return 1 if domain in URL_SHORTENERS else 0


def extract_features(url: str) -> dict:
    """
    Extract a fixed set of lexical/structural features from a URL string.

    Parameters
    ----------
    url : str
        The raw URL to analyse.

    Returns
    -------
    dict
        Feature name → numeric value mapping.
    """
    # Ensure scheme is present for urllib to parse correctly
    if not re.match(r"^https?://", url, re.IGNORECASE):
        url_to_parse = "http://" + url
    else:
        url_to_parse = url

    parsed = urllib.parse.urlparse(url_to_parse)
    extracted = tldextract.extract(url_to_parse)
    host = parsed.hostname or ""

    full_url = url_to_parse

    features = {
        # ── Length features ───────────────────────────────────────────────
        "url_length":            len(full_url),
        "domain_length":         len(host),
        "path_length":           len(parsed.path),

        # ── Character count features ──────────────────────────────────────
        "count_dots":            full_url.count("."),
        "count_hyphens":         full_url.count("-"),
        "count_underscores":     full_url.count("_"),
        "count_at":              full_url.count("@"),
        "count_digits":          sum(c.isdigit() for c in full_url),
        "count_slashes":         full_url.count("/"),
        "count_question_marks":  full_url.count("?"),
        "count_equals":          full_url.count("="),
        "count_ampersands":      full_url.count("&"),
        "count_percent":         full_url.count("%"),

        # ── Boolean / structural features ─────────────────────────────────
        "uses_https":            1 if parsed.scheme.lower() == "https" else 0,
        "is_ip_address":         _is_ip_address(host),
        "has_at_symbol":         1 if "@" in full_url else 0,
        "has_double_slash_redirect": _has_redirection(full_url),
        "uses_non_standard_port":   _uses_non_standard_port(parsed),
        "is_url_shortener":      _is_url_shortener(extracted),

        # ── Domain / subdomain features ───────────────────────────────────
        "num_subdomains":        _count_subdomains(extracted),

        # ── Ratio features ────────────────────────────────────────────────
        "digit_to_letter_ratio": _digit_to_letter_ratio(full_url),

        # ── Keyword features ──────────────────────────────────────────────
        "suspicious_keyword_count": _suspicious_keyword_count(full_url),
        "has_suspicious_keyword":   1 if _suspicious_keyword_count(full_url) > 0 else 0,
    }

    return features


def feature_names() -> list:
    """Return the ordered list of feature names (same order as extract_features)."""
    dummy = extract_features("http://example.com")
    return list(dummy.keys())


# ── Quick self-test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    test_urls = [
        "https://www.google.com",
        "http://192.168.1.1/login/verify?cmd=account&update=true",
        "http://bit.ly/3xABCDE",
        "https://paypal-secure-login.update-account.com/verify/billing",
        "http://amazon.com.phish-site.net/signin/confirm",
        "https://github.com/user/repo",
    ]

    print(f"{'URL':<60} {'Features':}")
    print("=" * 100)
    for url in test_urls:
        feats = extract_features(url)
        print(f"\n{url}")
        for k, v in feats.items():
            print(f"  {k:<35} {v}")
        print()
