import ipaddress
import socket
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

MAX_BYTES = 2_000_000
MAX_REDIRECTS = 3
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}


def validate_public_url(url):
    """Raise ValueError unless url is http(s) and resolves only to public IP addresses."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise ValueError("รองรับเฉพาะลิงก์ http/https")
    try:
        infos = socket.getaddrinfo(parsed.hostname, parsed.port or 443)
    except socket.gaierror as e:
        raise ValueError(f"ไม่พบโฮสต์: {parsed.hostname}") from e
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if not ip.is_global:
            raise ValueError("ไม่อนุญาตให้เข้าถึงที่อยู่ภายในหรือที่อยู่ส่วนตัว")


def _download(url):
    for _ in range(MAX_REDIRECTS + 1):
        validate_public_url(url)
        with requests.get(url, headers=HEADERS, timeout=10, stream=True, allow_redirects=False) as r:
            if r.is_redirect:
                url = urljoin(url, r.headers.get("Location", ""))
                continue
            r.raise_for_status()
            body = r.raw.read(MAX_BYTES + 1, decode_content=True)
            if len(body) > MAX_BYTES:
                raise ValueError("เนื้อหาใหญ่เกินไป")
            encoding = r.encoding if r.encoding and "charset" in r.headers.get("Content-Type", "") else None
            return body, encoding
    raise ValueError("redirect มากเกินไป")


def extract_text(html, encoding=None):
    """html may be str or bytes (bytes: encoding is auto-detected unless given)."""
    soup = BeautifulSoup(html, "html.parser", from_encoding=encoding if isinstance(html, bytes) else None)
    for element in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
        element.decompose()
    text = "\n".join(p.get_text().strip() for p in soup.find_all("p") if p.get_text().strip())
    return text or soup.get_text(separator="\n").strip()


def fetch_text_from_url(url):
    body, encoding = _download(url)
    return extract_text(body, encoding)


def decode_upload(data: bytes):
    for enc in ("utf-8-sig", "cp874"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("latin-1", errors="replace")
