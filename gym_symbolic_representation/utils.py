def clean_text(s, lower=False):
    txt = "".join(c if ord(c) < 128 else "-" for c in s)
    if lower:
        txt = txt.lower()
    return txt
