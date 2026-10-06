import logging
import os


logger = logging.getLogger(__name__)


def carregar_chave() -> None:
    arn = os.getenv("ANTHROPIC_API_KEY_SECRET_ARN", "").strip()
    if not arn or os.getenv("ANTHROPIC_API_KEY", "").strip():
        return
    try:
        import boto3

        resposta = boto3.client("secretsmanager").get_secret_value(SecretId=arn)
        chave = resposta.get("SecretString", "")
        if not isinstance(chave, str) or not chave.strip():
            raise ValueError("segredo vazio ou sem SecretString")
        os.environ["ANTHROPIC_API_KEY"] = chave
    except Exception:
        logger.warning("nao foi possivel carregar ANTHROPIC_API_KEY do Secrets Manager")
