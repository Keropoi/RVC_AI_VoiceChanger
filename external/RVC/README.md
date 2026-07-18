# Official RVC checkout

Clone or copy the exact official RVC repository used for training into this directory.
Keep its dependencies in `external/RVC/.venv`; the orchestration environment is `.venv`.
Run `python -m rvc_auto_trainer doctor` after adding it. The adapter inspects the checkout
at runtime and refuses to guess missing or incompatible scripts.
