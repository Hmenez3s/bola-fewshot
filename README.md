# bola-fewshot

Pesquisa: o FewShot-R18 remove falsos positivos de um detector de bola e melhora o detector sozinho?

Pergunta: dado o detector de bola `t1b_v5_etapa1_rodada1.pt` (`Evolvera/footlabs`, release `pesos-bola-20261125`), um filtro few-shot baseado no FewShot-R18 sobre os crops detectados melhora precisão/F1/mAP em relação ao detector sozinho, sem derrubar o recall?

O filtro atua exclusivamente em pós-detecção: avalia cada caixa proposta pelo detector e remove os candidatos abaixo do limiar congelado. Ele não detecta bolas novas e não recupera bolas perdidas.

## Desenho experimental (3 braços)

Mesmas imagens, mesmo detector (`t1b_v5_etapa1_rodada1.pt`), mesma resolução (`imgsz=1920`), mesma confiança (`conf=0.35`) e mesmo `IoU=0.50`:

1. **Braço 1 — Detector sozinho (`baseline`)**: `t1b_v5_etapa1_rodada1.pt` sem filtro.
2. **Braço 2 — Detector + FewShot-R18 de jogadores congelado (`frozen_players`, controle)**: checkpoint `six-references.pth` (`Hmenez3s/few-shot-team-goalkeeper-assignment`), treinado apenas para 4 classes de jogadores/goleiros, avaliado com protótipos few-shot (`k=1, 6`) em `layer4` (e variações controladas `layer4_pos_minus_neg`, `layer3`, `layer2` se a AUC de calibração ficar abaixo de `0.70`).
3. **Braço 3 — Detector + FewShot-R18-bola treinado (`trained_ball`)**: repete a receita episódica do paper (`train.py` / `six_references.py`): ResNet-18 inicializada do ImageNet (`fc = Identity`, apenas `layer3` e `layer4` treináveis, `BatchNorm2d` em `eval`), entrada letterbox RGB `48x96`, `AdamW(lr=2e-4, wd=1e-4)`, `CosineAnnealingLR`, 8 épocas × 180 episódios, seed `25`, `k_suporte=6` e `n_query=8` por classe (`{não-bola, bola}`), `logits = cosine * 20`, cross-entropy. Negativos minerados em `train.txt` com `conf >= 0.05` (e suplemento de fundo se faltar volume, com contagem declarada).

### Higiene de splits e protocolo sem vazamento

- Disjunção entre `train.txt` e `val.txt` verificada por caminho canônico (erro se houver sobreposição).
- Referências few-shot (`k=1, 6`) e treino episódico saem exclusivamente de `train.txt`.
- Como `val.txt` foi usado pelo detector para escolher o melhor checkpoint (não é teste virgem), `val.txt` é partido em duas metades fixas (50% `val_calib` / 50% `val_eval`, por jogo quando disponível): seleção de checkpoint, camada, `k` e limiar ocorre apenas em `val_calib`, congelada antes de medir `val_eval` uma única vez.
- Intervalos de confiança de 95% via bootstrap pareado (`2000` reamostragens) sobre imagens (e sobre jogos quando o mapa de jogos está presente) para `ΔF1`.
- Conjuntos virgens externos (`--extra-dataset`) são avaliados com protótipos e limiares congelados, sem reajuste.

## Configuração do detector

Parâmetros alinhados a `Evolvera/footlabs` (`server/engine/ball_detector.py`, `server/engine/cerco_beta/detector.py` e `server/engine/pipeline.py`):

- Classe única: `0` (`ball`).
- Leitura padrão do site para modelos T1b: `imgsz=1920`, `conf=0.35`, `iou=0.70`.
- O checkpoint `t1b_v5_etapa1_rodada1.pt` foi treinado com `train_args.imgsz=1280` (`--imgsz` configurável).

## Instalação e insumos

Pesos e datasets ficam fora do git (`.gitignore`).

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Baixar os pesos do detector (repo privado `Evolvera/footlabs`):

```bash
mkdir -p models
gh release download pesos-bola-20261125 -R Evolvera/footlabs -p t1b_v5_etapa1_rodada1.pt -D models
```

Opcional (segunda comparação com o modelo padrão do site):

```bash
gh release download pesos-bola-20261125 -R Evolvera/footlabs -p t1b_v4_etapa2_best.pt -D models
```

Obter o checkpoint de controle `six-references.pth` de `Hmenez3s/few-shot-team-goalkeeper-assignment` (ou apontar para a cópia local).

## Execução (um comando do detector ao relatório)

Com os dados copiados localmente em `/Users/j.henrique/bola-fewshot-data` (remapeando o prefixo Windows sem alterar os arquivos originais):

```bash
.venv/bin/python run_experiment.py \
  --data-yaml /Users/j.henrique/bola-fewshot-data/item488_t1b_v5_dataset_r5_9010/data.yaml \
  --remap-prefix C:/Footlabs_Miguel_cache=/Users/j.henrique/bola-fewshot-data \
  --detector-weights models/t1b_v5_etapa1_rodada1.pt \
  --fewshot-weights /Users/j.henrique/four-reference-team-classifier/models/six-references.pth \
  --imagenet-weights /Users/j.henrique/four-reference-team-classifier/data/inputs/resnet18.pth \
  --train-ball-arm \
  --imgsz 1920 \
  --conf 0.35 \
  --k-values 1 6 \
  --seed 42 \
  --train-seed 25 \
  --calib-source val_split \
  --calib-fraction 0.50 \
  --bootstrap-resamples 2000 \
  --device mps \
  --cache-dir .tmp/cache_v5_1920 \
  --output-dir results \
  --report-path RELATORIO.md
```

Opções adicionais úteis:
- `--group-map-json <arquivo.json>`: força partição 50/50 por jogo e adiciona bootstrap por jogo.
- `--extra-dataset NOME=CAMINHO/data.yaml` ou `--extra-val-list NOME=CAMINHO/lista.txt`: avalia conjuntos extras com protótipos e limiares congelados.

### Rodada 2 (CV aninhada de 5 dobras por jogo, contexto 2×/3×/4×, mineração no `train.txt` inteiro)

```bash
.venv/bin/python run_round2.py \
  --data-yaml /Users/j.henrique/bola-fewshot-data/item488_t1b_v5_dataset_r5_9010/data.yaml \
  --remap-prefix C:/Footlabs_Miguel_cache=/Users/j.henrique/bola-fewshot-data \
  --group-map-json .tmp/r5_9010_group_map.json \
  --detector-weights models/t1b_v5_etapa1_rodada1.pt \
  --fewshot-weights /Users/j.henrique/four-reference-team-classifier/models/six-references.pth \
  --imagenet-weights /Users/j.henrique/four-reference-team-classifier/data/inputs/resnet18.pth \
  --imgsz 1920 \
  --conf 0.35 \
  --context-scales 2 3 4 \
  --canvas 96x96 \
  --k-values 1 6 \
  --n-folds 5 \
  --seed 42 \
  --train-seed 25 \
  --train-epochs 16 \
  --train-steps-per-epoch 360 \
  --train-mining-conf 0.05 \
  --train-mining-imgsz 1280 \
  --bootstrap-resamples 2000 \
  --device mps \
  --cache-dir .tmp/r2_cache \
  --output-dir results
```

### Rodada 3 (few-shot por jogo com 6 auto-referências sem GT, recall alto `conf=0.15/0.10`, fusão com `conf` e baselines NMS/top-1)

```bash
.venv/bin/python run_round3.py \
  --data-yaml /Users/j.henrique/bola-fewshot-data/item488_t1b_v5_dataset_r5_9010/data.yaml \
  --remap-prefix C:/Footlabs_Miguel_cache=/Users/j.henrique/bola-fewshot-data \
  --group-map-json .tmp/r5_9010_group_map.json \
  --detector-weights models/t1b_v5_etapa1_rodada1.pt \
  --fewshot-weights /Users/j.henrique/four-reference-team-classifier/models/six-references.pth \
  --imagenet-weights /Users/j.henrique/four-reference-team-classifier/data/inputs/resnet18.pth \
  --imgsz 1920 \
  --operating-confs 0.35 0.15 0.10 \
  --context-scales 2 3 \
  --canvas 96x96 \
  --n-refs 6 \
  --ref-min-conf 0.60 \
  --n-folds 5 \
  --seed 42 \
  --train-seed 25 \
  --train-epochs 16 \
  --train-steps-per-epoch 360 \
  --train-mining-imgsz 1280 \
  --train-cap-per-game 80 \
  --bootstrap-resamples 2000 \
  --device mps \
  --cache-dir .tmp/r3_cache \
  --output-dir results
```

## Testes

```bash
.venv/bin/pytest -v
```
