"""
Input Validation
------------------
Addresses the Denial-of-Service gap identified in docs/ETHICS_SECURITY.md's
threat model: an oversized, malformed, or absurdly long audio file could hang
or crash the pipeline. This module validates a file BEFORE it reaches any of
the three scorers.

Every check here maps to a specific, realistic failure mode - not generic
"validation for its own sake."
"""

import os
from dataclasses import dataclass

ALLOWED_EXTENSIONS = {".wav", ".flac"}
MAX_FILE_SIZE_MB = 50          # a single call clip should never legitimately be larger
MAX_DURATION_SECONDS = 600     # 10 minutes - generous upper bound for a phone call clip


class AudioValidationError(Exception):
    """Raised when an audio file fails validation. Callers should catch this
    and reject the file rather than passing it to the scoring pipeline."""
    pass


@dataclass
class ValidationResult:
    valid: bool
    filepath: str
    size_mb: float
    duration_seconds: float
    reason: str = ""


def validate_audio_file(filepath: str) -> ValidationResult:
    """
    Runs all checks and returns a ValidationResult. Raises AudioValidationError
    on failure (fail closed - if you can't confirm it's safe, don't process it).

    Checks, in order (cheapest first, so we fail fast before doing expensive work):
      1. File exists
      2. Extension is an allowed audio format
      3. File size is within bounds (prevents memory exhaustion)
      4. Audio duration is within bounds (prevents pipeline hangs on huge clips)
    """
    if not os.path.isfile(filepath):
        raise AudioValidationError(f"File not found: {filepath}")

    ext = os.path.splitext(filepath)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise AudioValidationError(
            f"Unsupported file type '{ext}'. Allowed: {ALLOWED_EXTENSIONS}"
        )

    size_bytes = os.path.getsize(filepath)
    size_mb = size_bytes / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise AudioValidationError(
            f"File too large: {size_mb:.1f}MB exceeds {MAX_FILE_SIZE_MB}MB limit"
        )
    if size_bytes == 0:
        raise AudioValidationError("File is empty")

    # Duration check requires actually reading the file header - do this last
    # since it's the most expensive check.
    try:
        import soundfile as sf
        info = sf.info(filepath)
        duration = info.frames / info.samplerate
    except Exception as e:
        raise AudioValidationError(f"Could not read audio file header: {e}")

    if duration > MAX_DURATION_SECONDS:
        raise AudioValidationError(
            f"Audio too long: {duration:.0f}s exceeds {MAX_DURATION_SECONDS}s limit"
        )
    if duration <= 0:
        raise AudioValidationError("Audio file has zero or negative duration")

    return ValidationResult(
        valid=True,
        filepath=filepath,
        size_mb=size_mb,
        duration_seconds=duration,
    )


if __name__ == "__main__":
    # Smoke tests using deliberately bad inputs - no real audio file needed
    # to test the fast-fail checks.
    tests = [
        ("nonexistent.wav", "missing file"),
        ("test.mp3", "wrong extension"),
    ]

    # Create a dummy .mp3 to test the extension check
    with open("test.mp3", "wb") as f:
        f.write(b"not really audio")

    for filepath, description in tests:
        try:
            validate_audio_file(filepath)
            print(f"UNEXPECTED PASS: {description}")
        except AudioValidationError as e:
            print(f"Correctly rejected ({description}): {e}")

    os.remove("test.mp3")

    # Test empty file rejection
    with open("empty.wav", "wb") as f:
        pass
    try:
        validate_audio_file("empty.wav")
        print("UNEXPECTED PASS: empty file")
    except AudioValidationError as e:
        print(f"Correctly rejected (empty file): {e}")
    os.remove("empty.wav")
