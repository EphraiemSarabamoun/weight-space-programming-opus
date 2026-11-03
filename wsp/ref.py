"""Where the authors' released logs live (prompts, networks, answer keys).

Set WSP_REF to a clone of https://github.com/RikiyaT/lm_understand-logs (see fetch_ref.sh, which pins the commit
this reproduction used). Nothing in this repo copies the authors' prompts or networks; every experiment reads them
from there.
"""
import os

REF = os.path.expanduser(os.environ.get("WSP_REF", "~/projects/_ref/lm_understand-logs"))
