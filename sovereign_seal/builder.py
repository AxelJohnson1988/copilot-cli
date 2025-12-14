#!/usr/bin/env python3
"""
Sovereign Seal Builder (Encryptor)
==================================
Encrypts PDF documents using AES-256-GCM with Scrypt key derivation.

Output format: b"SS1" + salt(16 bytes) + nonce(12 bytes) + ciphertext
"""

import argparse
import getpass
import os
import sys
from pathlib import Path

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt
from cryptography.hazmat.backends import default_backend


# Sovereign Seal format constants
MAGIC_HEADER = b"SS1"
SALT_LENGTH = 16
NONCE_LENGTH = 12

# Scrypt parameters (n=2^15, r=8, p=1)
SCRYPT_N = 2**15  # 32768
SCRYPT_R = 8
SCRYPT_P = 1
KEY_LENGTH = 32  # 256 bits for AES-256


def derive_key(password: str, salt: bytes) -> bytes:
    """
    Derive a 32-byte key from password using Scrypt.

    Parameters:
        - n=2^15 (CPU/memory cost)
        - r=8 (block size)
        - p=1 (parallelization)
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


def encrypt_pdf(pdf_path: Path, password: str, output_path: Path) -> None:
    """
    Encrypt a PDF file and output as .seal format.

    Output structure: SS1 + salt(16) + nonce(12) + ciphertext
    """
    # Read the PDF file
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    pdf_data = pdf_path.read_bytes()

    if len(pdf_data) == 0:
        raise ValueError("PDF file is empty")

    print(f"[+] Read {len(pdf_data):,} bytes from {pdf_path.name}")

    # Generate cryptographic random salt and nonce
    salt = os.urandom(SALT_LENGTH)
    nonce = os.urandom(NONCE_LENGTH)

    # Derive key using Scrypt
    print("[+] Deriving encryption key with Scrypt (n=2^15, r=8, p=1)...")
    key = derive_key(password, salt)

    # Encrypt using AES-256-GCM
    print("[+] Encrypting with AES-256-GCM...")
    aesgcm = AESGCM(key)
    ciphertext = aesgcm.encrypt(nonce, pdf_data, associated_data=None)

    # Build the .seal file: MAGIC + SALT + NONCE + CIPHERTEXT
    seal_data = MAGIC_HEADER + salt + nonce + ciphertext

    # Write output
    output_path.write_bytes(seal_data)

    print(f"[+] Encrypted payload written to: {output_path}")
    print(f"    - Magic header: {MAGIC_HEADER.decode()}")
    print(f"    - Salt: {SALT_LENGTH} bytes")
    print(f"    - Nonce: {NONCE_LENGTH} bytes")
    print(f"    - Ciphertext: {len(ciphertext):,} bytes (includes 16-byte auth tag)")
    print(f"    - Total size: {len(seal_data):,} bytes")


def main():
    parser = argparse.ArgumentParser(
        description="Sovereign Seal Builder - Encrypt PDF documents for secure offline viewing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
    python builder.py patent.pdf
    python builder.py patent.pdf -o encrypted_patent.seal
    python builder.py patent.pdf --password mysecretpassword
        """
    )
    parser.add_argument(
        "pdf_file",
        type=Path,
        help="Path to the PDF file to encrypt"
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Output path for .seal file (default: payload.seal)"
    )
    parser.add_argument(
        "-p", "--password",
        type=str,
        default=None,
        help="Encryption password (will prompt if not provided)"
    )

    args = parser.parse_args()

    # Validate input file
    if not args.pdf_file.exists():
        print(f"[!] Error: File not found: {args.pdf_file}", file=sys.stderr)
        sys.exit(1)

    if not args.pdf_file.suffix.lower() == '.pdf':
        print(f"[!] Warning: File does not have .pdf extension: {args.pdf_file}")

    # Set default output path
    output_path = args.output if args.output else Path("payload.seal")

    # Get password
    if args.password:
        password = args.password
    else:
        password = getpass.getpass("[?] Enter encryption password: ")
        password_confirm = getpass.getpass("[?] Confirm encryption password: ")

        if password != password_confirm:
            print("[!] Error: Passwords do not match", file=sys.stderr)
            sys.exit(1)

    if len(password) < 8:
        print("[!] Warning: Password is less than 8 characters. Consider using a stronger password.")

    try:
        encrypt_pdf(args.pdf_file, password, output_path)
        print("\n[+] Encryption complete. Distribute payload.seal with the viewer executable.")
    except Exception as e:
        print(f"[!] Encryption failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
