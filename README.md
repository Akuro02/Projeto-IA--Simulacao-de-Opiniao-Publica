# Simulação de Opinião Pública com LLMs — CESOP/Unicamp

---

## 📋 Sumário de Entregas

| # | Entrega | Descrição | Arquivo |
|---|---------|-----------|---------|
| 1 | Artigo | Artigo completo em Markdown | `artigo_simulacao_cesop.md` |
| 2 | Notebook | Pipeline completo de simulação | `simulacao_opiniao_publica_cesop.ipynb` |
| 3 | Métricas | Resumo das métricas por questão e fold | `resumo_simulacao.csv` |
| 4 | Respostas simuladas | Respostas individuais geradas (fold 0) | `respostas_individuais_fold0.csv` |
| 5 | Figuras | Gráficos de distribuição e explicabilidade | `images/fig1.png` … `fig7.png` |
| 6 | Vídeo | Apresentação em vídeo (YouTube) | [link] |

---

## 🎯 Sobre o Projeto

Este projeto investiga a capacidade de **Modelos de Linguagem de Grande Escala (LLMs)** de simular respostas de questionários de opinião pública. Utilizando dados do CESOP/Unicamp sobre racismo ambiental e representação étnico-racial no Brasil (N = 2.000), três questões foram simuladas com o modelo `google/flan-t5-large` em configuração *zero-shot*.

---

## 📊 Questões Analisadas

| Código | Tema | Tipo | Distribuição empírica |
|--------|------|------|-----------------------|
| P12 | Conhecimento sobre *racismo ambiental* | Resposta única | Sim 24,3% / Não 75,7% |
| P20 | Espaços de representação étnico-racial adequada | Resposta múltipla | 11 opções |
| P25 | Conhecimento sobre *equidade* | Resposta única | Sim 27,0% / Não 73,0% |

---

## 🤖 Modelo e Configuração

| Item | Detalhe |
|------|---------|
| Modelo principal | `google/flan-t5-large` (770M parâmetros) |
| Configuração | Zero-shot, decodificação greedy |
| Modelo alternativo (GPU) | `mistralai/Mistral-7B-Instruct-v0.2` |
| Repetições (folds) | 3 × 200 respondentes |
| Sementes | {42, 137, 271} |

---

## 📈 Resultados Principais

| Questão | JSD | Interpretação |
|---------|-----|---------------|
| P25 — Equidade | 0,096 | Boa aderência |
| P12 — Racismo Ambiental | 0,160 | Aderência moderada |
| P20 — Representação (RM) | > 0,20 | Baixa aderência (alta entropia) |

- LLM superestima respostas "Sim" em P12 (+17,7 p.p.) e P25 (+10,5 p.p.)
- Random Forest treinado com pseudo-labels do LLM: F1-macro ≈ 0,50

---

## 🗂️ Estrutura do Repositório

```
📦 simulacao-opiniao-publica-cesop/
├── 📓 simulacao_opiniao_publica_cesop.ipynb   ← Notebook principal
├── Artigo
│    ├── 🖼️ images/
│    │   ├── fig1.png   ← Distribuições P12 e P25
│    │   ├── fig2.png   ← Distribuições P20
│    │   ├── fig3.png   ← Estabilidade entre folds
│    │   ├── fig4.png   ← Importância dos atributos (LLM)
│    │   ├── fig5.png   ← Análise por raça/cor × escolaridade
│    │   ├── fig6.png   ← Importância Gini (Random Forest)
│    │   └── fig7.png   ← Comparação LLM × Random Forest
│    └── 📄 artigo_simulacao_cesop.md               ← Artigo completo
└── 📝 README.md
```

---

## ▶️ Como Executar

1. Acesse [colab.research.google.com](https://colab.research.google.com)
2. Faça upload do arquivo `simulacao_opiniao_publica_cesop.ipynb`
3. Execute as células sequencialmente (`Runtime > Run all`)
4. **Nenhuma API key é necessária**

**Dependências:** `transformers` · `scikit-learn` · `pandas` · `numpy` · `matplotlib` · `seaborn` · `scipy`

**Hardware mínimo:** CPU (flan-t5) · GPU T4 recomendada para Mistral  
**Tempo estimado (CPU):** ~30–60 min para 200 respondentes × 3 folds

---

## 📚 Referências Principais

- ARGYLE et al. *Out of One, Many: Using Language Models to Simulate Human Samples.* Political Analysis, 2023.
- MIRANDA; BALBI. *Simulating Public Opinion: Comparing Distributional and Individual-Level Predictions from LLMs and Random Forests.* Entropy, v. 27, n. 923, 2025.
- CESOP/UNICAMP. *Pesquisa de Opinião Pública — Racismo Ambiental, Identidade Étnico-Racial e Desigualdades no Brasil*, 2023.

---

## 👥 Autores

> Autores: **Alex Kazuo Kodama - 10417942@mackenzista.com.br**, 
**Gabriel Tortolio Fonseca - 10416751@mackenzista.com.br**,
**Marco Antônio de Camargo - 10418309@mackenzista.com.br**,
**Natan Moreira Passos - 10417916@mackenzista.com.br**,
**Nícolas Henriques de Almeida - 10418357@mackenzista.com.br**,
**Thomas Pinheiro Grandin - 104181118@mackenzista.com.br**<br>

> Trabalho desenvolvido para a disciplina **Inteligência Artificial** — Ciência da Computação
  
> Universidade Presbiteriana Mackenzie — 7º semestre

