import unittest

from app.estrutura import aplicar_orcamento, cortar
from app.orcamento import (
    BULLETS_DEMAIS,
    BULLETS_RECENTES,
    BULLETS_TOTAL,
    COMPETENCIAS_MAX_TERMOS,
    PISO_BULLETS,
    RESUMO_MAX_FRASES,
    texto_orcamento,
)
from app.schemas import (
    CategoriaCompetencias,
    EstruturaCurriculo,
    ExperienciaEstruturada,
    FraseFonte,
    PerfilMestre,
    ReescritaEstruturada,
    TermoFonte,
)


def _perfil(n: int) -> PerfilMestre:
    return PerfilMestre.model_validate(
        {
            "nome": "Pessoa",
            "experiencias": [
                {"id": f"e{i}", "empresa": f"Empresa {i}", "cargo": "Cargo", "dataInicioAno": 2025 - 2 * i, "dataInicioMes": 1,
                 "dataFimAno": 2026 - 2 * i, "dataFimMes": 1, "atual": i == 0}
                for i in range(n)
            ],
        }
    )


def _estrutura(bullets_por_exp: list[int], resumo: int = 1) -> EstruturaCurriculo:
    return EstruturaCurriculo(
        resumo=[FraseFonte(texto=f"r{i}", fontes=["resumo"]) for i in range(resumo)],
        experiencias=[
            ExperienciaEstruturada(
                experiencia_id=f"e{i}",
                bullets=[FraseFonte(texto=f"e{i}b{j}", fontes=[f"e{i}"]) for j in range(n)],
            )
            for i, n in enumerate(bullets_por_exp)
        ],
    )


def _tamanhos(estrutura: EstruturaCurriculo) -> list[int]:
    return [len(item.bullets) for item in estrutura.experiencias]


class OrcamentoTest(unittest.TestCase):
    def test_excesso_e_cortado_no_fim_de_cada_lista(self):
        estrutura = aplicar_orcamento(_estrutura([6, 6, 6], resumo=5), _perfil(3))
        self.assertEqual(RESUMO_MAX_FRASES, len(estrutura.resumo))
        self.assertEqual([BULLETS_RECENTES, BULLETS_RECENTES, BULLETS_DEMAIS], _tamanhos(estrutura))
        self.assertEqual(["e0b0", "e0b1", "e0b2", "e0b3"], [b.texto for b in estrutura.experiencias[0].bullets])

    def test_total_respeita_uma_pagina_cortando_das_mais_antigas(self):
        estrutura = aplicar_orcamento(_estrutura([4, 4, 3, 3, 3]), _perfil(5))
        self.assertLessEqual(sum(_tamanhos(estrutura)), BULLETS_TOTAL)
        self.assertEqual(BULLETS_RECENTES, _tamanhos(estrutura)[0])
        self.assertTrue(all(n >= PISO_BULLETS for n in _tamanhos(estrutura)))

    def test_competencias_respeitam_o_teto_de_termos(self):
        estrutura = EstruturaCurriculo(
            competencias=[
                CategoriaCompetencias(categoria=f"C{c}", termos=[TermoFonte(termo=f"t{c}{i}", fonte="skills") for i in range(15)])
                for c in range(3)
            ]
        )
        cortada = aplicar_orcamento(estrutura, _perfil(1))
        self.assertEqual(COMPETENCIAS_MAX_TERMOS, sum(len(c.termos) for c in cortada.competencias))

    def test_orcamento_vai_na_descricao_do_schema_e_no_prompt_da_mesma_fonte(self):
        schema = ReescritaEstruturada.model_json_schema()
        bullets = schema["$defs"]["ExperienciaEstruturada"]["properties"]["bullets"]
        self.assertIn(f"no maximo {BULLETS_RECENTES}", bullets["description"])
        self.assertNotIn("maxItems", str(schema))
        self.assertIn(f"no maximo {BULLETS_TOTAL} bullets", texto_orcamento())


class CortePorNivelTest(unittest.TestCase):
    def test_cada_nivel_tira_o_ultimo_bullet_das_experiencias_mais_longas(self):
        estrutura = _estrutura([4, 4, 3])
        self.assertEqual([3, 3, 3], _tamanhos(cortar(estrutura, _perfil(3), 1)))
        self.assertEqual([2, 2, 2], _tamanhos(cortar(estrutura, _perfil(3), 2)))
        self.assertEqual(["e0b0", "e0b1"], [b.texto for b in cortar(estrutura, _perfil(3), 2).experiencias[0].bullets])
        self.assertEqual([4, 4, 3], _tamanhos(estrutura))

    def test_piso_de_dois_bullets_e_ate_tres_experiencias_nao_saem(self):
        cortada = cortar(_estrutura([2, 2, 2]), _perfil(3), 5)
        self.assertEqual([2, 2, 2], _tamanhos(cortada))
        self.assertEqual([], cortada.experiencias_omitidas)

    def test_com_mais_de_tres_sai_a_mais_antiga_e_nunca_a_mais_recente(self):
        cortada = cortar(_estrutura([2, 2, 2, 2, 2]), _perfil(5), 3)
        self.assertEqual(["e4", "e3"], cortada.experiencias_omitidas)
        self.assertNotIn("e0", cortada.experiencias_omitidas)

    def test_corte_e_deterministico(self):
        estrutura = _estrutura([4, 3, 3, 2])
        self.assertEqual(cortar(estrutura, _perfil(4), 2), cortar(estrutura, _perfil(4), 2))


if __name__ == "__main__":
    unittest.main()
