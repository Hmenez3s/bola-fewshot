# Relatório: Detector de bola + filtro de falsos positivos FewShot-R18

## Resumo (8 linhas)

1. Veredito (3 braços em `val_eval`, n=1623 imagens): FewShot-R18-bola treinado (trained_ball_layer4_pos_minus_neg_k6) deu ΔF1=-0.0220 (piora distinguível de zero); controle jogadores congelado (layer2_positive_only_k1) deu ΔF1=-0.0004 (indistinguível de zero).
2. Viés do split: val.txt não é teste virgem (usado na seleção do checkpoint do t1b_v5_etapa1_rodada1.pt; comparação relativa justa sob o mesmo viés, valores absolutos potencialmente otimistas); jogos/grupos: train=488, val=793, interseção=0 (por jogo); val dividido por grupo (642/151, interseção=0).
3. Métricas no teste final: Base P=0.7960/R=0.7815/F1=0.7887/mAP50=0.7251 | Jogadores congelado P=0.7984/R=0.7785/F1=0.7883/mAP50=0.7218 | Bola treinado P=0.8278/R=0.7139/F1=0.7667/mAP50=0.6685.
4. Custo real (trained_ball_layer4_pos_minus_neg_k6): removeu 85/329 FPs (25.8%) e removeu 111/1284 bolas verdadeiras por engano (ΔRecall=-0.0676).
5. Bootstrap pareado (2000 reamostragens): imagens IC 95% [-0.0309, -0.0129] (piora); jogos (n=151) IC 95% [-0.0622, -0.0096] (piora); treinado−congelado ΔF1=-0.0216 [-0.0310, -0.0129].
6. Separabilidade bola vs FP (sem limiar, no eval): jogadores congelado ROC-AUC=0.5125 (PR-AUC=0.8120); bola treinado ROC-AUC=0.6244 (PR-AUC=0.8554).
7. Viés de treino e teste virgem: taxa de FP em train=0.147/img vs val=0.203/img; teste virgem `soccernet_gsr_test_virgin` (n=490): ΔF1=-0.0123 [-0.0274, +0.0009], ROC-AUC=0.5456; box2boxcam não medido (inacessível localmente).
8. Teto de ganho possível do detector sozinho: 329 FPs em 1623 imagens (0.203 FP/img), teto de precisão=+0.2040 e teto de F1 sem novo recall=+0.0887; configurações testadas=12.

## Comparação principal no conjunto de teste final

Detector: `t1b_v5_etapa1_rodada1.pt` | Controle congelado: `six-references.pth` | imgsz=1920 | conf=0.35 | IoU=0.5 | Split final: `val_eval` (1623 imagens, 1643 bolas GT).

| Braço | Configuração | K | Limiar congelado | Precisão | Recall | F1 | mAP50 | mAP50-95 | FPs removidos | TPs removidos | ΔF1 (IC 95% imgs) | Distinguível de zero? |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Detector sozinho | baseline | - | - | 0.7960 | 0.7815 | 0.7887 | 0.7251 | 0.3577 | 0 / 329 | 0 / 1284 | 0.0000 | - |
| FewShot-R18 jogadores (congelado) | `layer4_positive_only_k1` | 1 | 0.5716 | 0.7959 | 0.7809 | 0.7883 | 0.7245 | 0.3575 | 0 / 329 | 1 / 1284 | -0.0004 [-0.0011, +0.0000] | Não (indistinguível) |
| FewShot-R18 jogadores (congelado) | `layer4_positive_only_k6` | 6 | -inf (sem corte) | 0.7960 | 0.7815 | 0.7887 | 0.7251 | 0.3577 | 0 / 329 | 0 / 1284 | +0.0000 [+0.0000, +0.0000] | Não (indistinguível) |
| FewShot-R18 jogadores (congelado) | `layer4_pos_minus_neg_k1` | 1 | -0.2767 | 0.7964 | 0.7809 | 0.7886 | 0.7247 | 0.3575 | 1 / 329 | 1 / 1284 | -0.0001 [-0.0011, +0.0007] | Não (indistinguível) |
| FewShot-R18 jogadores (congelado) | `layer4_pos_minus_neg_k6` | 6 | -0.1107 | 0.7965 | 0.7815 | 0.7889 | 0.7257 | 0.3580 | 1 / 329 | 0 / 1284 | +0.0002 [+0.0000, +0.0007] | Não (indistinguível) |
| FewShot-R18 jogadores (congelado) | `layer3_positive_only_k1` | 1 | 0.8632 | 0.7959 | 0.7809 | 0.7883 | 0.7244 | 0.3572 | 0 / 329 | 1 / 1284 | -0.0004 [-0.0011, +0.0000] | Não (indistinguível) |
| FewShot-R18 jogadores (congelado) | `layer3_positive_only_k6` | 6 | 0.8987 | 0.7955 | 0.7791 | 0.7872 | 0.7224 | 0.3556 | 0 / 329 | 4 / 1284 | -0.0015 [-0.0030, -0.0004] | Sim (piora) |
| FewShot-R18 jogadores (congelado) | `layer2_positive_only_k1` (ctrl, melhor-calib) | 1 | 0.9475 | 0.7984 | 0.7785 | 0.7883 | 0.7218 | 0.3551 | 6 / 329 | 5 / 1284 | -0.0004 [-0.0024, +0.0013] | Não (indistinguível) |
| FewShot-R18 jogadores (congelado) | `layer2_positive_only_k6` | 6 | 0.9600 | 0.7970 | 0.7766 | 0.7867 | 0.7201 | 0.3544 | 4 / 329 | 8 / 1284 | -0.0020 [-0.0043, -0.0002] | Sim (piora) |
| FewShot-R18-bola (treinado) | `trained_ball_layer4_pos_minus_neg_k1` | 1 | -0.0539 | 0.8278 | 0.7139 | 0.7667 | 0.6685 | 0.3362 | 85 / 329 | 111 / 1284 | -0.0220 [-0.0308, -0.0128] | Sim (piora) |
| FewShot-R18-bola (treinado) | `trained_ball_layer4_pos_minus_neg_k6` (bola) | 6 | -0.0151 | 0.8278 | 0.7139 | 0.7667 | 0.6685 | 0.3362 | 85 / 329 | 111 / 1284 | -0.0220 [-0.0309, -0.0129] | Sim (piora) |
| FewShot-R18-bola (treinado) | `trained_ball_layer4_positive_only_k1` | 1 | 0.9234 | 0.8280 | 0.7152 | 0.7675 | 0.6696 | 0.3364 | 85 / 329 | 109 / 1284 | -0.0212 [-0.0299, -0.0120] | Sim (piora) |
| FewShot-R18-bola (treinado) | `trained_ball_layer4_positive_only_k6` | 6 | 0.8574 | 0.7970 | 0.7815 | 0.7892 | 0.7251 | 0.3577 | 2 / 329 | 0 / 1284 | +0.0005 [+0.0000, +0.0012] | Não (indistinguível) |

Seleção de checkpoint, camada, K e limiar ocorreu exclusivamente em `val_calib`, congelada antes de medir `val_eval`.

## O FewShot-R18 separa bola de falso positivo?

| Braço | Configuração | Camada | Modo | K pos / neg | ROC-AUC (calib) | PR-AUC (calib) | ROC-AUC (eval) | PR-AUC (eval) | TP / FP (eval) |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| Detector sozinho (referência `conf`) | `detector_conf` | - | `conf` | - | 0.6934 | 0.8281 | 0.7786 | 0.9256 | 1284 / 329 |
| FewShot-R18 jogadores (congelado) | `layer4_positive_only_k1` | `layer4` | `positive_only` | 1 / 0 | 0.5837 | 0.7506 | 0.4393 | 0.7696 | 1284 / 329 |
| FewShot-R18 jogadores (congelado) | `layer4_positive_only_k6` | `layer4` | `positive_only` | 6 / 0 | 0.5757 | 0.7466 | 0.3752 | 0.7388 | 1284 / 329 |
| FewShot-R18 jogadores (congelado) | `layer4_pos_minus_neg_k1` | `layer4` | `pos_minus_neg` | 1 / 1 | 0.5490 | 0.7187 | 0.5740 | 0.8213 | 1284 / 329 |
| FewShot-R18 jogadores (congelado) | `layer4_pos_minus_neg_k6` | `layer4` | `pos_minus_neg` | 6 / 6 | 0.5202 | 0.7011 | 0.5660 | 0.8234 | 1284 / 329 |
| FewShot-R18 jogadores (congelado) | `layer3_positive_only_k1` | `layer3` | `positive_only` | 1 / 0 | 0.6708 | 0.7941 | 0.4396 | 0.7830 | 1284 / 329 |
| FewShot-R18 jogadores (congelado) | `layer3_positive_only_k6` | `layer3` | `positive_only` | 6 / 0 | 0.6451 | 0.7759 | 0.4131 | 0.7575 | 1284 / 329 |
| FewShot-R18 jogadores (congelado) | `layer2_positive_only_k1` | `layer2` | `positive_only` | 1 / 0 | 0.6502 | 0.7778 | 0.5125 | 0.8120 | 1284 / 329 |
| FewShot-R18 jogadores (congelado) | `layer2_positive_only_k6` | `layer2` | `positive_only` | 6 / 0 | 0.6319 | 0.7679 | 0.4504 | 0.7760 | 1284 / 329 |
| FewShot-R18-bola (treinado) | `trained_ball_layer4_pos_minus_neg_k1` | `layer4` | `pos_minus_neg` | 1 / 1 | 0.7040 | 0.8239 | 0.6244 | 0.8554 | 1284 / 329 |
| FewShot-R18-bola (treinado) | `trained_ball_layer4_pos_minus_neg_k6` | `layer4` | `pos_minus_neg` | 6 / 6 | 0.7042 | 0.8240 | 0.6244 | 0.8554 | 1284 / 329 |
| FewShot-R18-bola (treinado) | `trained_ball_layer4_positive_only_k1` | `layer4` | `positive_only` | 1 / 0 | 0.6773 | 0.7939 | 0.6143 | 0.8437 | 1284 / 329 |
| FewShot-R18-bola (treinado) | `trained_ball_layer4_positive_only_k6` | `layer4` | `positive_only` | 6 / 0 | 0.5263 | 0.7257 | 0.5432 | 0.8212 | 1284 / 329 |

## Treino episódico do FewShot-R18-bola (receita do paper)

- Arquitetura e otimização: ResNet-18 ImageNet (`fc=Identity`, apenas `layer3` e `layer4` treináveis, `BatchNorm2d` em `eval`), entrada letterbox RGB `[48, 96]`, `AdamW(lr=0.0002, wd=0.0001)`, `CosineAnnealingLR`, `8` épocas × `180` episódios (`k_suporte=6`, `n_query=8` por classe, `cosine * 20.0`, seed `25`).
- Dispositivo e velocidade: `mps`, 1º episódio `0.4368s`, tempo total `119.83s`, melhor época selecionada em `val_calib`: época `4`.
- Mineração em `train.txt` (`1500` imagens varridas com `conf>=0.05`): `1489` recortes positivos, `220` FPs de alta confiança (`>=0.35`), `2145` FPs minerados em baixa confiança (`0.05..0.35`), `0` recortes suplementares de fundo.
- Viés de treino do detector (`conf>=0.35`): taxa de FP em `train` = `0.1467` FP/img (`220` FPs, precisão `0.8306`) vs em `val_eval` = `0.2027` FP/img (`329` FPs, precisão `0.7960`).

| Época | Loss treino | ROC-AUC (`val_calib`) | PR-AUC (`val_calib`) | Melhor F1 (`val_calib`) | Tempo (s) |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.5710 | 0.6854 | 0.8083 | 0.6451 | 9.4 |
| 2 | 0.5189 | 0.6892 | 0.7952 | 0.6444 | 9.1 |
| 3 | 0.4816 | 0.6956 | 0.8122 | 0.6460 | 15.1 |
| 4 * | 0.4079 | 0.7042 | 0.8240 | 0.6442 | 16.8 |
| 5 | 0.3776 | 0.6702 | 0.7955 | 0.6443 | 17.2 |
| 6 | 0.3270 | 0.6953 | 0.8110 | 0.6446 | 17.6 |
| 7 | 0.2789 | 0.6964 | 0.8085 | 0.6451 | 17.2 |
| 8 | 0.2633 | 0.6907 | 0.8054 | 0.6450 | 17.1 |

## Teto de ganho possível (headroom)

- Detecções base no eval: 1613 (1284 TP, 329 FP, 359 FN; 0.203 FP/imagem).
- Precisão base: 0.7960 → teto teórico com filtro perfeito: 1.0000 (ganho máximo de precisão: +0.2040).
- F1 base: 0.7887 → teto teórico sem novo recall: 0.8773 (ganho máximo de F1: +0.0887).

## Protocolo, higiene de splits e escolhas

- Fonte dos splits: `/Users/j.henrique/bola-fewshot-data/item488_t1b_v5_dataset_r5_9010/data.yaml` (lista de validação: `data.yaml:val`; disjunção `train`/`val` verificada: `True`).
- Referências do filtro: sorteadas automaticamente apenas de `train` (`train_ref`: 32852 imagens, 32719 bolas GT) com seed fixa `42`.
- Calibração do limiar: `val_split` (`val_calib`, 1624 imagens, fração `0.5`). Limiar congelado antes de tocar `val_eval` (1623 imagens).
- Pré-processamento do crop: canvas letterbox RGB `[48, 96]`, `context_scale=1.0`, `square_crop=False`.
- Configurações testadas: 12 (política de variações `always`, gatilho AUC `< 0.7`, variações executadas: `True`).
- Escolha geral na calibração: `layer2_positive_only_k1` — selected strictly on 'val_calib' by highest calibration F1 (0.6452 vs baseline 0.6381), tie-broken by recall and calib ROC-AUC; threshold frozen before opening 'val_eval'.
- Auditoria por jogo/grupo: `train`=488 grupos, `val`=793 grupos, interseção `train∩val`=0 (split por jogo); `val_calib`=642 grupos vs `val_eval`=151 grupos (interseção=0).
- Status do conjunto de validação: val.txt não é teste virgem (usado na seleção do checkpoint do t1b_v5_etapa1_rodada1.pt; comparação relativa justa sob o mesmo viés, valores absolutos potencialmente otimistas).

## Conjuntos extras / teste virgem fora de r1

### soccernet_gsr_test_virgin

Imagens: 490 | Bolas GT: 456 | Protótipos e limiares congelados (sem reajuste neste conjunto).

| Braço | Configuração congelada | Limiar | Precisão | Recall | F1 | mAP50 | FPs removidos | TPs removidos | ROC-AUC | ΔF1 (IC 95%) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Detector sozinho | baseline | - | 0.5308 | 0.4342 | 0.4777 | 0.2701 | 0 / 175 | 0 / 198 | 0.5814 | 0.0000 |
| FewShot-R18 jogadores (congelado) | `layer2_positive_only_k1` | 0.9475 | 0.5135 | 0.3750 | 0.4335 | 0.2268 | 13 / 175 | 27 / 198 | 0.3884 | -0.0442 [-0.0636, -0.0254] |
| FewShot-R18-bola (treinado) | `trained_ball_layer4_pos_minus_neg_k6` | -0.0151 | 0.5457 | 0.4057 | 0.4654 | 0.2545 | 21 / 175 | 13 / 198 | 0.5456 | -0.0123 [-0.0274, +0.0009] |

- `box2boxcam` (câmera amadora/noturna, 198 quadros): não medido por não estar disponível localmente.

## Limitações

- O conjunto `val.txt` do `r5_9010` não é teste virgem do detector: foi usado durante o treino do `t1b_v5_etapa1_rodada1.pt` para selecionar o melhor checkpoint.
- O filtro atua apenas em pós-processamento (remoção de FPs) e nunca aumenta recall nem recupera bolas não propostas pelo detector.
- Bolas em quadros 1080p frequentemente medem ~10×10 pixels; o redimensionamento para 48×96 letterbox pode perder detalhes finos e depende fortemente da aparência local.
- A mineração de negativos em `train.txt` sofre do viés de treino do detector (menos FPs de alta confiança do que fora do treino), exigindo mineração em confiança baixa (`0.05..0.35`).
- O conjunto amador noturno `box2boxcam` (198 quadros) não estava disponível localmente para medição direta.
