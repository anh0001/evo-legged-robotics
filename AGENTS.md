---
# Tell Codex to run submodule init BEFORE indexing files
pre:
  - git submodule update --init --recursive
---