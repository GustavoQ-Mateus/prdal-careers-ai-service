import os
import sys


if os.getenv("AWS_LAMBDA_RUNTIME_API"):
    os.execvp(sys.executable, [sys.executable, "-m", "awslambdaric", *sys.argv[1:]])

os.execvp("uvicorn", [
    "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000",
    "--timeout-graceful-shutdown", "25",
])
