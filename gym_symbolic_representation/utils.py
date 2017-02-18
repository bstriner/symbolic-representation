def clean_text(s):
    return "".join(c if ord(c) < 128 else "-" for c in s)