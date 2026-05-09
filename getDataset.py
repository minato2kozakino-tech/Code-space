import kagglehub

# Download latest version
path = kagglehub.dataset_download("starblasters8/human-vs-llm-text-corpus")

print("Path to dataset files:", path)