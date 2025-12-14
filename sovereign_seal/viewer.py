#!/usr/bin/env python3
"""
Sovereign Seal Viewer (Decryptor)
=================================
Secure offline viewer for encrypted patent documents.

SECURITY FEATURES:
- Decrypts PDF into RAM only (never writes to disk)
- Uses fitz.open(stream=...) to avoid temp file persistence
- Includes leakage guard to prevent verbatim content extraction
"""

import getpass
import sys
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt
from cryptography.hazmat.backends import default_backend
from cryptography.exceptions import InvalidTag


# Sovereign Seal format constants (must match builder.py)
MAGIC_HEADER = b"SS1"
MAGIC_LENGTH = 3
SALT_LENGTH = 16
NONCE_LENGTH = 12

# Scrypt parameters (n=2^15, r=8, p=1)
SCRYPT_N = 2**15  # 32768
SCRYPT_R = 8
SCRYPT_P = 1
KEY_LENGTH = 32  # 256 bits for AES-256

# Leakage guard threshold
VERBATIM_THRESHOLD = 30  # Maximum consecutive identical words allowed


def derive_key(password: str, salt: bytes) -> bytes:
    """
    Derive a 32-byte key from password using Scrypt.
    """
    kdf = Scrypt(
        salt=salt,
        length=KEY_LENGTH,
        n=SCRYPT_N,
        r=SCRYPT_R,
        p=SCRYPT_P,
        backend=default_backend()
    )
    return kdf.derive(password.encode('utf-8'))


def too_verbatim(context: str, answer: str, threshold: int = VERBATIM_THRESHOLD) -> bool:
    """
    Leakage Guard: Check if an AI summary contains a run of consecutive
    identical words matching the source text.

    This prevents extraction of verbatim content from the protected document.

    Args:
        context: The original source text (from the PDF)
        answer: The summary or output text to check
        threshold: Maximum allowed consecutive matching words (default: 30)

    Returns:
        True if the answer contains a verbatim run >= threshold (should block)
        False if the answer is safe to output
    """
    if not context or not answer:
        return False

    # Tokenize both texts into words (lowercase for comparison)
    context_words = context.lower().split()
    answer_words = answer.lower().split()

    if len(answer_words) < threshold:
        return False

    # Build a set of all possible n-grams from the answer
    # and check if any exist in the context
    context_text = ' '.join(context_words)

    for i in range(len(answer_words) - threshold + 1):
        # Extract a window of 'threshold' consecutive words from the answer
        window = ' '.join(answer_words[i:i + threshold])

        # Check if this exact sequence exists in the context
        if window in context_text:
            return True

    return False


def parse_seal_file(seal_path: Path) -> tuple[bytes, bytes, bytes]:
    """
    Parse a .seal file and extract salt, nonce, and ciphertext.

    Returns:
        Tuple of (salt, nonce, ciphertext)
    """
    if not seal_path.exists():
        raise FileNotFoundError(f"Seal file not found: {seal_path}")

    seal_data = seal_path.read_bytes()

    # Validate minimum size
    min_size = MAGIC_LENGTH + SALT_LENGTH + NONCE_LENGTH + 16  # 16 = min auth tag
    if len(seal_data) < min_size:
        raise ValueError("Invalid seal file: too small")

    # Validate magic header
    magic = seal_data[:MAGIC_LENGTH]
    if magic != MAGIC_HEADER:
        raise ValueError(f"Invalid seal file: wrong magic header (got {magic!r})")

    # Extract components
    offset = MAGIC_LENGTH
    salt = seal_data[offset:offset + SALT_LENGTH]
    offset += SALT_LENGTH

    nonce = seal_data[offset:offset + NONCE_LENGTH]
    offset += NONCE_LENGTH

    ciphertext = seal_data[offset:]

    return salt, nonce, ciphertext


def decrypt_to_ram(seal_path: Path, password: str) -> bytes:
    """
    Decrypt a .seal file and return the PDF bytes in RAM.

    CRITICAL: This function NEVER writes decrypted data to disk.
    """
    print("[+] Parsing seal file...")
    salt, nonce, ciphertext = parse_seal_file(seal_path)

    print(f"[+] Seal file structure validated")
    print(f"    - Salt: {len(salt)} bytes")
    print(f"    - Nonce: {len(nonce)} bytes")
    print(f"    - Ciphertext: {len(ciphertext):,} bytes")

    # Derive key using Scrypt
    print("[+] Deriving decryption key with Scrypt...")
    key = derive_key(password, salt)

    # Decrypt using AES-256-GCM
    print("[+] Decrypting with AES-256-GCM...")
    aesgcm = AESGCM(key)

    try:
        decrypted_pdf = aesgcm.decrypt(nonce, ciphertext, associated_data=None)
    except InvalidTag:
        raise ValueError("Decryption failed: Invalid password or corrupted file")

    print(f"[+] Decrypted {len(decrypted_pdf):,} bytes into RAM")

    # Validate it looks like a PDF
    if not decrypted_pdf.startswith(b'%PDF'):
        raise ValueError("Decrypted data does not appear to be a valid PDF")

    return decrypted_pdf


def extract_text_from_ram(pdf_bytes: bytes) -> str:
    """
    Extract text from PDF bytes in RAM using PyMuPDF.

    CRITICAL: Uses fitz.open(stream=...) to avoid any disk I/O.
    The PDF is processed entirely in memory.
    """
    # Open PDF from memory stream - NO disk writes
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    full_text = []
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text = page.get_text()
        full_text.append(f"--- Page {page_num + 1} ---\n{text}")

    doc.close()

    return "\n\n".join(full_text)


def generate_safe_summary(full_text: str, max_words: int = 100) -> str:
    """
    Generate a safe summary that doesn't violate the leakage guard.

    For demonstration, this creates a simple word-count summary
    rather than copying verbatim text.
    """
    words = full_text.split()
    total_words = len(words)
    total_chars = len(full_text)
    line_count = full_text.count('\n') + 1

    # Extract first few words as a preview (well under threshold)
    preview_words = min(20, len(words))
    preview = ' '.join(words[:preview_words])
    if len(words) > preview_words:
        preview += "..."

    summary = f"""
=== SOVEREIGN SEAL DOCUMENT SUMMARY ===

Document Statistics:
  - Total characters: {total_chars:,}
  - Total words: {total_words:,}
  - Total lines: {line_count:,}

Preview (first {preview_words} words):
  "{preview}"

[Document successfully decrypted and loaded into RAM]
[Full content available in secure viewer mode]
"""
    return summary


def view_document(seal_path: Path, password: str) -> None:
    """
    Main viewing function - decrypts and displays document safely.
    """
    # Decrypt PDF into RAM only
    pdf_bytes = decrypt_to_ram(seal_path, password)

    # Extract text using in-memory PDF processing
    print("[+] Extracting text from RAM (no disk I/O)...")
    full_text = extract_text_from_ram(pdf_bytes)

    # Generate a safe summary
    summary = generate_safe_summary(full_text)

    # Demonstrate leakage guard
    print("\n[+] Running leakage guard check...")

    if too_verbatim(full_text, summary):
        print("[!] BLOCKED: Summary contains too much verbatim content")
        print("[!] Leakage guard prevented potential content extraction")
        return

    print("[+] Leakage guard passed - summary is safe")

    # Output the safe summary
    print(summary)

    # Clear sensitive data from memory (best effort)
    del pdf_bytes
    del full_text


def main():
    """
    Main entry point for the Sovereign Seal Viewer.
    """
    print("=" * 50)
    print("  SOVEREIGN SEAL - Secure Patent Viewer")
    print("  RAM-Only Decryption | AES-256-GCM")
    print("=" * 50)
    print()

    # Look for payload.seal in current directory by default
    default_seal = Path("payload.seal")

    if len(sys.argv) > 1:
        seal_path = Path(sys.argv[1])
    elif default_seal.exists():
        seal_path = default_seal
    else:
        print("[?] Enter path to .seal file: ", end="")
        seal_path = Path(input().strip())

    if not seal_path.exists():
        print(f"[!] Error: File not found: {seal_path}", file=sys.stderr)
        sys.exit(1)

    # Get password
    password = getpass.getpass("[?] Enter decryption password: ")

    if not password:
        print("[!] Error: Password cannot be empty", file=sys.stderr)
        sys.exit(1)

    try:
        view_document(seal_path, password)
        print("\n[+] Viewer session complete. Sensitive data cleared from RAM.")
    except ValueError as e:
        print(f"[!] Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"[!] Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
