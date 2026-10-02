---
id: reescrita
versao: 1
---

# Modo Pipeline de Curriculo

Sistema generico de analise ATS e geracao de curriculos otimizados para um produto multiusuario. A fonte factual de cada geracao e sempre o perfil-mestre e o contexto do usuario autenticado enviados no request. Este spec define metodologia, nao fatos pessoais.

## Regra de ouro

Manter apenas informacoes verdadeiras. Nao inventar metricas, datas, empresas, tecnologias, senioridade, resultados, dados pessoais ou autoria individual. Nao transferir fatos entre experiencias. Uma keyword da vaga so pode entrar no curriculo se existir no perfil-mestre ou no contexto factual daquele usuario.

## ETAPA 1 - Analise ATS

Analisar o curriculo/base factual contra a vaga como um filtro automatizado faria. Retornar:

- Score ATS estimado X/100.
- Keywords da vaga encontradas.
- Keywords criticas ausentes.
- Pontos que eliminam ou reduzem aderencia.
- Veredicto em texto curto.

## ETAPA 2 - Reescrita otimizada

Reescrever o curriculo completo com:

- Titulo profissional alinhado a vaga.
- Resumo profissional com as principais keywords factuais da descricao da vaga.
- Competencias logo apos o resumo, com categorias ATS-friendly.
- No maximo 3 experiencias, priorizadas por aderencia a vaga.
- Experiencias em ordem cronologica reversa, a mais recente primeiro. A experiencia atual (sem data de termino) nunca pode ficar depois de uma ja encerrada.
- 2 a 4 bullets densos por experiencia.
- Pagina unica.

Formula obrigatoria de bullet ATS:

Verbo de acao -> o que foi feito -> resultado real -> ferramenta por extenso.

Padrao de densidade:

- Bullets substanciais com contexto, tecnologia concreta e escala/impacto real.
- Nunca usar uma linha generica que apenas liste tarefa.
- Nao inventar numeros. Usar numeros somente quando existirem no perfil/contexto.
- Preservar moldura de autoria: contribuicao de time continua colaborativa; autoria forte so quando estiver comprovada na propria experiencia.

Espelhamento de keywords:

- Usar termos exatos da vaga quando forem sustentados pela fonte factual do usuario.
- Meta de overlap de vocabulario com a vaga >= 60%.
- Sem keyword stuffing.
- Sem secao "palavras-chave".

## Estrutura ATS obrigatoria do Markdown

Usar exatamente esta ordem, traduzindo os titulos de secao quando o idioma da vaga exigir:

1. Nome.
2. Titulo profissional.
3. Linha de contato no corpo.
4. `## RESUMO PROFISSIONAL`
5. `## COMPETENCIAS`
6. `## EXPERIENCIA PROFISSIONAL`
7. `## FORMACAO ACADEMICA`
8. `## CERTIFICACOES`
9. `## IDIOMAS`

Regras de texto:

- Datas em MM/AAAA quando existirem.
- Sem tabelas, colunas ou icones.
- Sem travessao Unicode.
- Separadores simples com `|` ou virgula.
- Markdown limpo e ATS-friendly.

## ETAPA 3 - ATS pos-geracao

Medir o score ATS do curriculo gerado e comparar com a Etapa 1. Se o score final ficar abaixo do limiar por keywords ausentes que existem na fonte factual do usuario, executar um passe dirigido de reescrita antes de retornar.

## Degradacao

Se a geracao por modelo falhar, violar contrato ou introduzir tecnologia ausente da fonte factual do usuario, rejeitar a saida e tentar novamente com erro explicito. Se persistir, retornar fallback deterministico factual com degradacao visivel para a UI.
