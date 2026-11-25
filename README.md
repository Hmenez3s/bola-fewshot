# bola-fewshot

Pesquisa: o FewShot-R18 remove falsos positivos de um detector de bola e melhora o detector sozinho?

Pergunta: dado o detector de bola `t1b_v5_etapa1_rodada1.pt` (`Evolvera/footlabs`, release `pesos-bola-20261125`), um filtro few-shot com o FewShot-R18 sobre os crops detectados melhora precisão/F1/mAP em relação ao detector sozinho, sem derrubar o recall?

O FewShot-R18 atua exclusivamente como filtro pós-detecção: avalia cada caixa proposta pelo detector e remove os candidatos abaixo do limiar congelado. Ele não detecta bolas novas e não recupera bolas perdidas.

## Desenho do filtro (decisão de projeto)

O FewShot-R18 (`Hmenez3s/few-shot-team-goalkeeper-assignment`, checkpoint `six-references.pth`) é uma ResNet-18 (`fc = Identity`, embedding de 512 dimensões) com `layer3` e `layer4` ajustadas para separar 4 classes de jogadores/goleiros usando crops RGB em letterbox `48x96`. Ele **não** foi treinado para bola.

O protocolo implementado testa a separabilidade antes de qualquer limiar:

1. **Hipótese principal (`layer4_positive_only`)**: sorteia `k` crops de bolas ground-truth apenas do split de treino (semente fixa), extrai embeddings L2-normalizados da `layer4` (512-d), calcula o protótipo médio normalizado e mede similaridade de cosseno para cada detecção. Mede `ROC-AUC` e `PR-AUC` (TP vs FP) antes de escolher o limiar.
2. **Variações controladas (`--variations auto`, ativadas se a melhor `calib_roc_auc` principal ficar abaixo de `0.70`, ou com `--variations always`)**:
   - `layer4_pos_minus_neg`: adiciona protótipo negativo a partir de falsos positivos do detector no próprio treino (`cos_pos - cos_neg`).
   - `layer3_positive_only`: embedding intermediário `layer3` (256-d, GAP).
   - `layer2_positive_only`: embedding `layer2` (128-d, GAP — pesos congelados do ImageNet no treino original do FewShot-R18).
3. **Higiene de splits e limiar congelado**:
   - Disjunção entre `train` e `val` verificada por caminho canônico (falha se houver sobreposição).
   - Referências (`k=1, 6`) saem exclusivamente de `train`.
   - O limiar de corte é escolhido na calibração (`val_split`: corte fixo 30%/70% do `val` declarado antes da avaliação, ou `train_holdout`) maximizando F1 e é congelado antes de abrir o conjunto final.
   - Todas as configurações testadas ficam registradas em `results/experiment_results.json` e `RELATORIO.md`, com intervalo bootstrap pareado de 95% sobre imagens para `ΔF1`.

## Configuração do detector

Parâmetros alinhados a `Evolvera/footlabs` (`server/engine/ball_detector.py`, `server/engine/cerco_beta/detector.py` e `server/engine/pipeline.py`):

- Classe única: `0` (`ball`).
- Leitura padrão do site para modelos T1b: `imgsz=1920`, `conf=0.35`, `iou=0.70`.
- Observação do checkpoint `t1b_v5_etapa1_rodada1.pt`: `train_args.imgsz=1280` (use `--imgsz 1280` se quiser reproduzir o tamanho de treino em vez da leitura do site em `1920`).

## Instalação e insumos

Os pesos e o dataset ficam fora do git (`.gitignore`).

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

Opcional (segunda comparação com o peso padrão do site):

```bash
gh release download pesos-bola-20261125 -R Evolvera/footlabs -p t1b_v4_etapa2_best.pt -D models
```

Baixar o peso do FewShot-R18 (`six-references.pth`) do repositório `Hmenez3s/few-shot-team-goalkeeper-assignment` (ou apontar para a cópia local `models/six-references.pth`).

## Execução (um comando do detector ao relatório)

No Windows (Dense, somente leitura no dataset, saída em pasta própria `C:\bola_fewshot_run`):

```powershell
python run_experiment.py `
  --data-yaml C:\Footlabs_Miguel_cache\item488_t1b_v5_dataset_r5_9010\data.yaml `
  --images-root C:\Footlabs_Miguel_cache\item488_t1b_v5_dataset_r1\images `
  --labels-root C:\Footlabs_Miguel_cache\item488_t1b_v5_dataset_r1\labels `
  --detector-weights models\t1b_v5_etapa1_rodada1.pt `
  --fewshot-weights models\six-references.pth `
  --imgsz 1920 `
  --conf 0.35 `
  --k-values 1 6 `
  --seed 42 `
  --calib-source val_split `
  --calib-fraction 0.30 `
  --bootstrap-resamples 2000 `
  --cache-dir C:\bola_fewshot_run\cache `
  --output-dir results `
  --report-path RELATORIO.md
```

Se o conjunto de câmera amadora (`box2boxcam`) estiver acessível, adicione:

```text
--extra-dataset box2boxcam=CAMINHO\PARA\box2boxcam\data.yaml
```

O comando aplica o filtro e o limiar já congelados sobre `box2boxcam`, sem reajustar parâmetros, e grava `results/experiment_results.json` e `RELATORIO.md`.

## Testes

```bash
.venv/bin/pytest -v
```
