import importlib.util
import unittest

from app import rag


@unittest.skipUnless(importlib.util.find_spec("sentence_transformers"), "modelo de embedding nao instalado")
class ModeloRealTest(unittest.TestCase):
    def test_teto_do_chunk_e_o_limite_do_modelo_e_comporta_o_chunk_inteiro(self):
        self.assertEqual(rag.limite_tokens(), rag._model().max_seq_length)
        self.assertGreaterEqual(rag.limite_tokens(), 512)
        rag.conferir_dimensao()


if __name__ == "__main__":
    unittest.main()
