"""
Encryption at Rest
---------------------
Addresses the Information Disclosure gap identified in docs/ETHICS_SECURITY.md -
call transcripts can contain PII and financial details. If call_log.db is ever
copied off the machine (stolen laptop, misconfigured backup), transcripts
should not be readable as plaintext.

Uses Fernet (symmetric, authenticated encryption) from the `cryptography`
library - the standard, well-audited choice for "encrypt a field before
storing it locally" use cases like this one. We are NOT implementing our own
crypto - we're using a vetted library correctly, which is the right call for
a student project (rolling your own encryption is a well-known anti-pattern).

Install: pip install cryptography
"""

import os
from cryptography.fernet import Fernet, InvalidToken

KEY_FILE = os.path.join(os.path.dirname(__file__), ".encryption_key")


def load_or_create_key() -> bytes:
    """
    Loads the encryption key from a local file, generating one on first run.

    IMPORTANT SECURITY NOTE FOR YOUR REPORT: storing the key alongside the
    code (even in a gitignored file) means anyone with filesystem access to
    the machine can decrypt the data - this protects against the specific
    threat we identified (a copied/stolen DATABASE FILE alone), not against a
    fully compromised machine. This is an honest, documented limitation, not
    a claim of end-to-end security. A production system would use an OS
    keychain or a secrets manager (e.g. HashiCorp Vault) instead - noted as
    future work.
    """
    if os.path.exists(KEY_FILE):
        with open(KEY_FILE, "rb") as f:
            return f.read()

    key = Fernet.generate_key()
    with open(KEY_FILE, "wb") as f:
        f.write(key)
    # Restrict file permissions to owner-only where the OS supports it
    # (no-op on Windows, works on macOS/Linux)
    try:
        os.chmod(KEY_FILE, 0o600)
    except OSError:
        pass
    return key


def encrypt_text(plaintext: str) -> str:
    """Encrypts a string, returning a base64-encoded ciphertext safe to store as TEXT in SQLite."""
    if not plaintext:
        return ""
    key = load_or_create_key()
    fernet = Fernet(key)
    return fernet.encrypt(plaintext.encode("utf-8")).decode("utf-8")


def decrypt_text(ciphertext: str) -> str:
    """Decrypts a string previously produced by encrypt_text(). Raises InvalidToken if the
    ciphertext is corrupted or the key doesn't match."""
    if not ciphertext:
        return ""
    key = load_or_create_key()
    fernet = Fernet(key)
    try:
        return fernet.decrypt(ciphertext.encode("utf-8")).decode("utf-8")
    except InvalidToken:
        raise ValueError(
            "Failed to decrypt - wrong key, or the ciphertext has been tampered with"
        )


if __name__ == "__main__":
    original = "This is urgent, wire transfer the funds immediately."

    encrypted = encrypt_text(original)
    print(f"Original:  {original}")
    print(f"Encrypted: {encrypted[:50]}... (not human-readable)")

    decrypted = decrypt_text(encrypted)
    print(f"Decrypted: {decrypted}")

    assert decrypted == original, "Round-trip encryption/decryption failed!"
    print("\nRound-trip test PASSED: encrypt -> store -> decrypt returns original text.")

    # Demonstrate tamper detection
    tampered = encrypted[:-5] + "XXXXX"
    try:
        decrypt_text(tampered)
        print("UNEXPECTED: tampered ciphertext was accepted!")
    except ValueError as e:
        print(f"Tamper detection PASSED: {e}")

    os.remove(KEY_FILE)  # cleanup after test
