"""

import kagglehub

# Download latest version
path = kagglehub.dataset_download("starblasters8/human-vs-llm-text-corpus")

print("Path to dataset files:", path)
"""

# Compatibility shim: some versions of kagglesdk renamed `get_web_endpoint` to `get_endpoint`.
# Ensure the older name exists before importing `kagglehub` which may expect it.
try:
	import kagglesdk.kaggle_env as _kenv
	if not hasattr(_kenv, 'get_web_endpoint') and hasattr(_kenv, 'get_endpoint'):
		_kenv.get_web_endpoint = _kenv.get_endpoint
except Exception:
	# If kagglesdk is not installed or another error occurs, we'll let kagglehub import fail and surface the error.
	pass

import kagglehub

# Download latest version
path = kagglehub.dataset_download("thedevastator/evol-instruct-code-80k-v1-dataset")

print("Path to dataset files:", path)