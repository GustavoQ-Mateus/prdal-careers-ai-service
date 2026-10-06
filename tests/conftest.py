import os

os.environ["PRDAL_AMBIENTE"] = "desenvolvimento"
os.environ.pop("SERVICE_TOKEN", None)
os.environ["AI_JUIZ_RELACAO"] = "0"
os.environ.pop("ANTHROPIC_API_KEY_SECRET_ARN", None)
