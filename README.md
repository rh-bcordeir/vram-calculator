# Quanta VRAM

Uma calculadora de dimensionamento de GPU para inferência de LLM no vLLM. Cole um repositório
do Hugging Face, defina a sua carga de trabalho e veja se o modelo cabe — separado em pesos,
cache KV e overhead, comparado com a memória que o motor de inferência tem permissão de usar
de verdade.

Feita com React e Vite. Sem backend, sem dados embutidos no build, sem rastreamento.

## Por quê

A regra de bolso que todo mundo repete — mais ou menos 2 bytes por parâmetro em precisão de
16 bits — cobre só os pesos. Em produção o cache KV costuma ser o número maior, e ele cresce
com o tamanho do contexto _e_ com a concorrência. Um modelo que carrega numa boa ainda vai cair
sob carga se a piscina de cache secar.

Esta ferramenta calcula os três componentes e também resolve a equação ao contrário: quantas
requisições simultâneas cabem no seu contexto, e quanto contexto cabe na sua concorrência.

## Como rodar

```bash
npm install
npm run dev      # http://localhost:5173
npm run build    # saída estática em dist/
npm run preview  # serve o build de produção localmente
```

## De onde vêm os números

```
pesos          = (params − kept) × bytes_por_param + kept × kept_bytes
kv_por_camada  = elems × replicas × kv_bytes          por token
kv_total       = kv_por_camada × tokens_em_cache × concorrência
overhead       = 0,59 GiB por dispositivo + batched_tokens × hidden × 13,35 × 2
utilizável     = capacidade_gpu × qtd_gpus × gpu_memory_utilization
```

- **`kept`** é a parte do modelo que a quantização não toca — tabela de embeddings, cabeça de
  saída, roteadores de MoE, torres de visão. São 3% do Llama 3.3 70B, mas 12% do Kimi K2, então
  ignorar isso subestima feio um checkpoint MoE quantizado. `kept_bytes` é a largura dessa
  parte: 2 quase sempre, 4 nos modelos publicados em float32.
- **`elems`** é `2 × kv_heads × head_dim` na atenção padrão — o 2 cobre o tensor de chave e o de
  valor, e `kv_heads` significa `num_key_value_heads`, não `num_attention_heads`. Modelos de
  atenção latente guardam um único vetor comprimido no lugar disso, duas ordens de grandeza
  menor.
- **`replicas`** é quantas cópias do cache o paralelismo de tensores acaba mantendo. O vLLM
  distribui as cabeças KV entre as GPUs enquanto houver cabeças suficientes e replica quando
  não houver: um modelo de 2 cabeças KV em 8 GPUs fica com 4 cópias, e um cache latente de
  cabeça única fica com 8.
- **`tokens_em_cache`** conta só as camadas que realmente guardam cache KV — 4 das 40 do
  Granite 4.0 H, já que as outras são Mamba — e para de cobrar pelas camadas com janela
  deslizante assim que o contexto passa da janela delas.
- **Overhead** é uma reta ajustada a partir de duas medições em um cluster real; a memória de
  ativação acompanha os tokens em voo e a largura do modelo, não a quantidade de pesos. O
  `CALIBRATION.md` mostra a conta inteira.

As capacidades dos aceleradores são os valores que o driver informa, não o número de marketing:
uma A100 de "80GB" tem cerca de 79,2 GiB.

## Conferindo contra os mínimos publicados pela Red Hat

```bash
npm run validate              # resumo, mais qualquer linha fora de ±10%
npm run validate -- --verbose # todas as 61 linhas
```

Todo modelo da lista de presets é um modelo que a Red Hat valida, e 61 dessas linhas publicam
uma vRAM mínima e uma lista de configurações de GPU suportadas. O
`scripts/redhat-minimums.json` guarda essa tabela; o script confere o tamanho **e** o formato —
que cada máquina publicada seja uma que o app deixa você escolher, e que ela comporte os pesos.
As 61 ficam dentro de 10% do valor publicado, 58 dentro de 5%, sem nenhuma falha de formato.

## Consulta ao Hugging Face

Nada é buscado ao abrir a página. Quando você aperta **Load**, saem duas requisições:

| Requisição | O que fornece |
| --- | --- |
| `/{repo}/resolve/main/config.json` | camadas, cabeças KV, head dim, rank latente, mistura de camadas, quantização, contexto treinado |
| `/api/models/{repo}` | contagem de parâmetros e a fatia em 16 bits, dos metadados `safetensors` |

Os repositórios da própria Mistral trazem `params.json` em vez de `config.json`; esse formato
também é lido.

Cada repositório é lido no máximo uma vez por sessão e fica num cache em memória.

O `config.json` é lido com desconfiança, porque os campos nem sempre estão onde se espera:

- Repositórios multimodais aninham a config do modelo de linguagem em `text_config`
- `head_dim` falta com frequência e é derivado de `hidden_size ÷ num_attention_heads`
- A ausência de `num_key_value_heads` significa MHA puro, então cai de volta para
  `num_attention_heads`
- `quantization_config` define a precisão dos pesos automaticamente quando existe, lendo o
  `num_bits` por grupo que o compressed-tensors aninha, em vez do nível de cima
- `kv_lora_rank` + `qk_rope_head_dim` marcam um modelo de atenção latente e dimensionam o cache
  real dele
- As camadas de atenção são contadas a partir de `layer_types`, do
  `hybrid_override_pattern` da Nemotron ou do `full_attention_interval` do Qwen3 Next — o que a
  config publicar
- Um `sliding_window` é ignorado quando `use_sliding_window` é falso, que é como o Qwen2.5 e o
  Phi-4 Mini declaram uma janela que não usam

Todo campo continua editável, então uma consulta que falha nunca te trava.

### Quando a fórmula não se aplica

O app levanta um aviso em vez de devolver um número errado em silêncio:

- **Mixture of Experts** — todo peso de especialista ocupa memória, não só os ativos
- **Atenção latente (MLA)** — modelada, incluindo a cópia que cada GPU mantém
- **Arquiteturas híbridas** — só as camadas de atenção guardam cache KV; o estado de Mamba ou
  de atenção linear é pequeno, fixo, e não é contado aqui
- **Atenção com janela deslizante** — modelada camada a camada, já que a maioria desses modelos
  aplica a janela só em parte delas

### Repositórios com acesso restrito e CORS

O navegador chama o `huggingface.co` diretamente, o que funciona para repositórios públicos.
Dois casos precisam de proxy:

- O repositório é restrito ou privado e exige token
- A sua rede bloqueia a requisição cross-origin

O `vite.config.js` já inclui um proxy `/hf` de desenvolvimento para o segundo caso — basta
definir `VITE_HF_BASE=/hf`. Para repositórios restritos em produção, coloque um proxy pequeno
na frente que injete o token no servidor e aponte `VITE_HF_BASE` para ele. Nunca mande um token
do Hugging Face para o navegador.

## Trate o resultado como estimativa

Confirme na realidade antes de provisionar qualquer coisa. O vLLM imprime `GPU KV cache size`
na inicialização, depois do profiling, e isso diz exatamente quantos tokens de cache sobraram.
Num cluster rodando, `vllm:gpu_cache_usage_perc` e `vllm:num_preemptions_total` mostram se o
dimensionamento se sustentou sob tráfego real — preempções significam cache sob pressão.

A diferença entre a estimativa e o valor observado é um fator de calibração estável para aquele
runtime e aquele acelerador. Meça uma vez e aplique depois.

## Estrutura

```
scripts/
  redhat-minimums.json  a tabela de vRAM publicada, transcrita do PDF
  validate-redhat.mjs   confere cada preset contra ela
src/
  lib/
    constants.js     aceleradores, precisões, presets
    vram.js          o cálculo, como função pura
    huggingface.js   parsing de repositório, extração de config, cache de fetch
    useTheme.js      claro/escuro, persistido, definido antes do primeiro paint
  components/
    ModelPanel.jsx   consulta de repositório e campos do modelo
    MemoryMap.jsx    a barra de alocação
    Fields.jsx       controles de formulário compartilhados
  App.jsx
  index.css          os dois temas como CSS custom properties
```

O `vram.js` não tem React nem I/O, então dá para levar para uma CLI ou um notebook sem mudar
nada.

## Licença

MIT
