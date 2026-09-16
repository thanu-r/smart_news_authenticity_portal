import socket
import ipaddress
import urllib.parse
from typing import Tuple, Optional
import requests
import logging

logger = logging.getLogger("smartnews.safe_fetcher")

MAX_CONTENT_LENGTH = 5 * 1024 * 1024  # 5 MB
DEFAULT_TIMEOUT = 8.0  # seconds
MAX_REDIRECTS = 5

class SSRFSecurityError(ValueError):
    """Raised when a URL targets a private, loopback, or disallowed network address."""
    pass

class SafeFetchError(Exception):
    """Raised when fetching a URL fails or exceeds safety limits."""
    pass

def is_ip_allowed(ip_str: str) -> bool:
    """
    Check whether an IP address is a safe, publicly routable address.
    Blocks private (RFC 1918), loopback, link-local (169.254.x.x / fe80::),
    reserved, multicast, and unspecified addresses.
    """
    try:
        ip = ipaddress.ip_address(ip_str)
    except ValueError:
        return False

    if (
        ip.is_loopback or
        ip.is_private or
        ip.is_link_local or
        ip.is_reserved or
        ip.is_multicast or
        ip.is_unspecified
    ):
        return False

    # Block AWS / GCP / Azure link-local metadata service (169.254.169.254)
    if str(ip) == "169.254.169.254":
        return False

    return True

def validate_url_safety(url: str) -> Tuple[str, str, int]:
    """
    Validate that the URL uses http/https and does not resolve to an internal/private IP.
    Returns (cleaned_url, hostname, port).
    Raises SSRFSecurityError if unsafe.
    """
    parsed = urllib.parse.urlparse(url.strip())
    if parsed.scheme.lower() not in ("http", "https"):
        raise SSRFSecurityError(f"Unsupported protocol scheme '{parsed.scheme}'. Only HTTP and HTTPS are permitted.")

    hostname = parsed.hostname
    if not hostname:
        raise SSRFSecurityError("URL does not specify a valid hostname.")

    # Block explicit localhost representations
    lower_host = hostname.lower().strip(".")
    if lower_host in ("localhost", "local", "ip6-localhost", "ip6-loopback"):
        raise SSRFSecurityError(f"Access to localhost/internal domain '{hostname}' is blocked for security.")

    # Determine port
    port = parsed.port or (443 if parsed.scheme.lower() == "https" else 80)

    # Resolve hostname via DNS to verify all destination IPs
    try:
        addr_info = socket.getaddrinfo(hostname, port, proto=socket.IPPROTO_TCP)
    except socket.gaierror as e:
        raise SafeFetchError(f"Could not resolve domain name '{hostname}': {str(e)}")

    resolved_ips = set()
    for entry in addr_info:
        ip_addr = entry[4][0]
        resolved_ips.add(ip_addr)

    if not resolved_ips:
        raise SafeFetchError(f"No IP addresses resolved for hostname '{hostname}'.")

    for ip in resolved_ips:
        if not is_ip_allowed(ip):
            logger.warning(f"SSRF violation prevented: '{url}' resolved to disallowed IP {ip}")
            raise SSRFSecurityError(
                f"Security policy violation: Hostname '{hostname}' resolves to restricted internal/private IP ({ip})."
            )

    return url.strip(), hostname, port

def safe_fetch_url(url: str, timeout: float = DEFAULT_TIMEOUT) -> Tuple[str, str, int]:
    """
    Safely fetch content from a user-supplied URL with SSRF protection,
    safe redirect inspection, timeout, and response size limiting.
    
    Returns:
      (content_text, final_url, status_code)
    """
    current_url = url
    session = requests.Session()
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36 (SmartNewsVerifier/2.0)"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    }

    for redirect_count in range(MAX_REDIRECTS + 1):
        # Validate safety of the current URL / redirect target
        validate_url_safety(current_url)

        try:
            resp = session.get(
                current_url,
                headers=headers,
                timeout=timeout,
                allow_redirects=False,
                stream=True
            )
        except requests.exceptions.Timeout:
            raise SafeFetchError(f"Connection timed out after {timeout}s while attempting to fetch URL.")
        except requests.exceptions.RequestException as e:
            raise SafeFetchError(f"Network error while fetching URL: {str(e)}")

        # Handle redirects manually to inspect every new location
        if resp.status_code in (301, 302, 303, 307, 308):
            location = resp.headers.get("Location")
            if not location:
                raise SafeFetchError(f"Redirect response {resp.status_code} missing Location header.")

            # Resolve relative redirects
            current_url = urllib.parse.urljoin(current_url, location)
            logger.info(f"Following safe redirect ({redirect_count + 1}/{MAX_REDIRECTS}) -> {current_url}")
            resp.close()
            continue

        # Check response status
        if resp.status_code != 200:
            resp.close()
            raise SafeFetchError(f"Web server returned HTTP error status {resp.status_code}.")

        # Read content safely with size limit
        content_bytes = bytearray()
        for chunk in resp.iter_content(chunk_size=65536):
            if chunk:
                content_bytes.extend(chunk)
                if len(content_bytes) > MAX_CONTENT_LENGTH:
                    resp.close()
                    raise SafeFetchError(f"Content length exceeds maximum allowed limit of {MAX_CONTENT_LENGTH // (1024 * 1024)}MB.")

        # Decode content with detected or fallback encoding
        encoding = resp.encoding or "utf-8"
        try:
            text = content_bytes.decode(encoding, errors="replace")
        except Exception:
            text = content_bytes.decode("utf-8", errors="replace")

        return text, current_url, resp.status_code

    raise SafeFetchError(f"Too many redirects (exceeded maximum limit of {MAX_REDIRECTS}).")
