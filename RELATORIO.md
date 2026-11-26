# Relatório: Detector de bola + filtro FewShot-R18 (Rodada 1 e Rodada 2)

## Resumo (8 linhas)

1. Veredito (Rodada 1 vs Rodada 2): A conclusão MUDOU na Rodada 2: com contexto ampliado, CV de 5 dobras por jogo e treino completo em train.txt, FewShot-R18-bola (treinado layer3/4) obteve melhora distinguível de zero (ΔF1=+0.0066, contra ΔF1=-0.0220 na Rodada 1).
2. Correções aplicadas na Rodada 2: recorte quadrado `96x96` com `context_scale` ∈ `[2.0, 3.0, 4.0]` escolhido só na calibração (`2×` escolhido em 4/5 dobras, `3×` em 1/5), CV aninhada de 5 dobras por jogo sobre todo o `val.txt` (3247 imagens, 793 jogos; `train∩val=0`), mineração em todo o `train.txt` (32852 imagens) e 5760 episódios.
3. Métricas OOF na Rodada 2 (todo `val.txt`, imgsz=1920, conf=0.35): Base P=0.7535/R=0.6811/F1=0.7155/mAP50=0.6032 | Congelado P=0.7548/R=0.6796/F1=0.7152/mAP50=0.6019 | Treinado layer3/4 P=0.7838/R=0.6694/F1=0.7221/mAP50=0.6002 | Treinado todas camadas P=0.7716/R=0.6676/F1=0.7158/mAP50=0.5982.
4. Custo real (FewShot-R18-bola (treinado layer3/4)): removeu 127/742 FPs (17.1%; sendo 110 FPs distantes `IoU<0.10`) e removeu 39/2268 bolas verdadeiras por engano (ΔRecall=-0.0117).
5. Bootstrap pareado (2000 reamostragens) para `trained_ball_layer34` vs Base: jogos (n=793) ΔF1=+0.0066 IC 95% [+0.0011, +0.0124] (melhora); imagens (n=3247) IC 95% [+0.0029, +0.0103]; média das 5 dobras=+0.0056±0.0074; supera também o controle congelado (ΔF1=+0.0068, IC jogos [+0.0011, +0.0130]).
6. Separabilidade OOF bola vs FP (ROC-AUC): `conf` do detector YOLO=0.7395; jogadores congelado média/dobra=0.5159; bola treinado layer3/4 média/dobra=0.6598 (melhor config fixa OOF=0.7131 em `ctx3`); todas camadas média/dobra=0.6296 (colapsou no treino).
7. Rótulo ruidoso (`results/amostra_fp.png`, 40 FPs em conf>=0.50): 9/40 (22.5%, IC Wilson [12.3%, 37.5%]) são bolas extras não anotadas (+2 duvidosos) e 22/40 (55,0%) são quase-acertos/duplicados da bola anotada; descontando bolas não anotadas, ΔF1 estimado=-0.0019. `soccernet_gsr_test_virgin` descartado como prova (caixas GT deslocadas).
8. Viés e teto de ganho: `val.txt` não é teste virgem do detector (usado na escolha do checkpoint; comparação relativa justa, absolutos possivelmente otimistas); FP/img `train`=0.0554 vs `val`=0.2285; headroom base em `val`=+0.2465 em precisão e +0.0948 em F1; configurações testadas: Rodada 1=12, Rodada 2=36 fixas + 3 seletores CV aninhados.

## Rodada 2 — Validação cruzada aninhada (5 dobras por jogo, contexto 2×/3×/4×, treino completo)

Detector: `t1b_v5_etapa1_rodada1.pt` (`imgsz=1920`, `conf=0.35`, `IoU=0.5`) | Conjunto avaliado: todo o `val.txt` (3247 imagens, 3330 bolas GT, 793 jogos; `train∩val=0` jogos). Aviso de viés: `val.txt` foi usado pelo detector para escolher o melhor checkpoint (não é teste virgem; comparação entre braços sob o mesmo viés).

### Comparação OOF agregada dos braços (seleção aninhada de contexto, K, época e limiar por dobra)

| Braço | Precisão | Recall | F1 | mAP50 | mAP50-95 | FPs removidos | TPs removidos | ROC-AUC (média 5 dobras) | ΔF1 (IC 95% jogos) | ΔF1 (IC 95% imgs) | Média±DP (5 dobras) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Detector sozinho (baseline) | 0.7535 | 0.6811 | 0.7155 | 0.6032 | 0.2875 | 0 / 742 | 0 / 2268 | 0.7395 (`conf`) | 0.0000 | 0.0000 | 0.0000 |
| Referência: subir `conf` do YOLO (CV 5 dobras) | 0.7723 | 0.6721 | 0.7187 | 0.5964 | 0.2849 | 82 / 742 | 30 / 2268 | 0.7395 | +0.0032 [-0.0023, +0.0061] | [+0.0003, +0.0061] | +0.0018 ± 0.0028 |
| FewShot-R18 jogadores (congelado) | 0.7548 | 0.6796 | 0.7152 | 0.6019 | 0.2871 | 7 / 742 | 5 / 2268 | 0.5159 | -0.0002 [-0.0020, +0.0007] | [-0.0012, +0.0006] | -0.0005 ± 0.0014 |
| FewShot-R18-bola (treinado layer3/4) | 0.7838 | 0.6694 | 0.7221 | 0.6002 | 0.2863 | 127 / 742 | 39 / 2268 | 0.6598 | +0.0066 [+0.0011, +0.0124] | [+0.0029, +0.0103] | +0.0056 ± 0.0074 |
| FewShot-R18-bola (treinado todas camadas) | 0.7716 | 0.6676 | 0.7158 | 0.5982 | 0.2838 | 83 / 742 | 46 / 2268 | 0.6296 | +0.0004 [-0.0067, +0.0040] | [-0.0030, +0.0035] | -0.0015 ± 0.0060 |

### Escolhas congeladas por dobra (sem olhar a dobra de teste)

#### FewShot-R18 jogadores (congelado)

| Dobra | Imagens | Config escolhida (calib) | Época | Limiar congelado | AUC calib | AUC dobra | F1 base dobra | F1 filt dobra | ΔF1 dobra | FPs rem | TPs rem |
|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1111 | `frozen_ctx4_pos_minus_neg_k1` | 0 | -0.2810 | 0.4967 | 0.5064 | 0.8548 | 0.8548 | +0.0000 | 0 | 0 |
| 1 | 534 | `frozen_ctx2_positive_only_k1` | 0 | 0.3880 | 0.5114 | 0.4786 | 0.6155 | 0.6125 | -0.0030 | 2 | 3 |
| 2 | 534 | `frozen_ctx2_positive_only_k1` | 0 | 0.3739 | 0.5029 | 0.5477 | 0.6545 | 0.6545 | -0.0000 | 2 | 1 |
| 3 | 534 | `frozen_ctx4_pos_minus_neg_k1` | 0 | -0.2810 | 0.5093 | 0.4814 | 0.6073 | 0.6073 | +0.0000 | 0 | 0 |
| 4 | 534 | `frozen_ctx2_positive_only_k1` | 0 | 0.3717 | 0.5018 | 0.5656 | 0.6564 | 0.6570 | +0.0006 | 3 | 1 |
- Decomposição `v14` vs resto: `v14` (1111 imgs) F1 base=0.8548 → filt=0.8548 (Δ=+0.0000); demais 792 jogos (2136 imgs) F1 base=0.6330 → filt=0.6324 (Δ=-0.0006).

#### FewShot-R18-bola (treinado layer3/4)

| Dobra | Imagens | Config escolhida (calib) | Época | Limiar congelado | AUC calib | AUC dobra | F1 base dobra | F1 filt dobra | ΔF1 dobra | FPs rem | TPs rem |
|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1111 | `trained_ball_layer34_ctx2_positive_only_k6` | 13 | 0.8491 | 0.6241 | 0.6207 | 0.8548 | 0.8565 | +0.0017 | 10 | 4 |
| 1 | 534 | `trained_ball_layer34_ctx2_positive_only_k1` | 16 | 0.6905 | 0.6973 | 0.6778 | 0.6155 | 0.6245 | +0.0089 | 25 | 5 |
| 2 | 534 | `trained_ball_layer34_ctx2_pos_minus_neg_k6` | 16 | -0.1561 | 0.6946 | 0.6926 | 0.6545 | 0.6590 | +0.0044 | 27 | 10 |
| 3 | 534 | `trained_ball_layer34_ctx2_positive_only_k6` | 13 | 0.8491 | 0.6183 | 0.6447 | 0.6073 | 0.6236 | +0.0163 | 45 | 8 |
| 4 | 534 | `trained_ball_layer34_ctx3_pos_minus_neg_k6` | 8 | -0.1540 | 0.7208 | 0.6631 | 0.6564 | 0.6532 | -0.0032 | 20 | 12 |
- Decomposição `v14` vs resto: `v14` (1111 imgs) F1 base=0.8548 → filt=0.8565 (Δ=+0.0017); demais 792 jogos (2136 imgs) F1 base=0.6330 → filt=0.6399 (Δ=+0.0068).

#### FewShot-R18-bola (treinado todas camadas)

| Dobra | Imagens | Config escolhida (calib) | Época | Limiar congelado | AUC calib | AUC dobra | F1 base dobra | F1 filt dobra | ΔF1 dobra | FPs rem | TPs rem |
|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1111 | `trained_ball_all_layers_ctx3_pos_minus_neg_k6` | 15 | -0.0000 | 0.6322 | 0.5484 | 0.8548 | 0.8544 | -0.0005 | 0 | 1 |
| 1 | 534 | `trained_ball_all_layers_ctx4_positive_only_k1` | 1 | 0.9921 | 0.6176 | 0.6788 | 0.6155 | 0.6218 | +0.0063 | 32 | 10 |
| 2 | 534 | `trained_ball_all_layers_ctx4_positive_only_k1` | 1 | 0.9925 | 0.6257 | 0.6365 | 0.6545 | 0.6456 | -0.0089 | 22 | 17 |
| 3 | 534 | `trained_ball_all_layers_ctx2_positive_only_k6` | 1 | 0.9915 | 0.5921 | 0.6922 | 0.6073 | 0.6086 | +0.0013 | 16 | 6 |
| 4 | 534 | `trained_ball_all_layers_ctx2_pos_minus_neg_k1` | 1 | -0.0009 | 0.6165 | 0.5921 | 0.6564 | 0.6505 | -0.0059 | 13 | 12 |
- Decomposição `v14` vs resto: `v14` (1111 imgs) F1 base=0.8548 → filt=0.8544 (Δ=-0.0005); demais 792 jogos (2136 imgs) F1 base=0.6330 → filt=0.6312 (Δ=-0.0018).

### Comparação direta entre braços treinados e controle congelado (bootstrap pareado OOF)

- `trained_ball_layer34_minus_frozen_players`: ΔF1=+0.0068 | IC 95% jogos [+0.0011, +0.0130] (melhora) | IC 95% imagens [+0.0030, +0.0107] (melhora).
- `trained_ball_all_layers_minus_frozen_players`: ΔF1=+0.0006 | IC 95% jogos [-0.0066, +0.0043] (indistinguível) | IC 95% imagens [-0.0029, +0.0039] (indistinguível).

### Todas as configurações fixas testadas na Rodada 2 (OOF 5 dobras)

| Braço | Configuração | ROC-AUC OOF | PR-AUC OOF | Precisão | Recall | F1 | mAP50 | FPs rem | TPs rem | ΔF1 (IC 95% jogos) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FewShot-R18 jogadores (congelado) | `frozen_ctx2_pos_minus_neg_k1` | 0.5518 | 0.7679 | 0.7543 | 0.6805 | 0.7155 | 0.6028 | 4/742 | 2/2268 | +0.0000 [-0.0010, +0.0008] |
| FewShot-R18 jogadores (congelado) | `frozen_ctx2_pos_minus_neg_k6` | 0.5645 | 0.7660 | 0.7537 | 0.6802 | 0.7151 | 0.6026 | 2/742 | 3/2268 | -0.0004 [-0.0015, +0.0001] |
| FewShot-R18 jogadores (congelado) | `frozen_ctx2_positive_only_k1` | 0.5115 | 0.7501 | 0.7560 | 0.6793 | 0.7156 | 0.6022 | 12/742 | 6/2268 | +0.0001 [-0.0016, +0.0015] |
| FewShot-R18 jogadores (congelado) | `frozen_ctx2_positive_only_k6` | 0.5399 | 0.7663 | 0.7532 | 0.6793 | 0.7144 | 0.6020 | 1/742 | 6/2268 | -0.0011 [-0.0036, +0.0000] |
| FewShot-R18 jogadores (congelado) | `frozen_ctx3_pos_minus_neg_k1` | 0.5235 | 0.7491 | 0.7538 | 0.6796 | 0.7148 | 0.6018 | 2/742 | 6/2268 | -0.0007 [-0.0026, +0.0002] |
| FewShot-R18 jogadores (congelado) | `frozen_ctx3_pos_minus_neg_k6` | 0.5306 | 0.7587 | 0.7534 | 0.6790 | 0.7143 | 0.6009 | 2/742 | 7/2268 | -0.0012 [-0.0039, -0.0000] |
| FewShot-R18 jogadores (congelado) | `frozen_ctx3_positive_only_k1` | 0.4385 | 0.7113 | 0.7535 | 0.6811 | 0.7155 | 0.6032 | 0/742 | 0/2268 | +0.0000 [+0.0000, +0.0000] |
| FewShot-R18 jogadores (congelado) | `frozen_ctx3_positive_only_k6` | 0.5336 | 0.7490 | 0.7536 | 0.6796 | 0.7147 | 0.6020 | 2/742 | 5/2268 | -0.0008 [-0.0032, +0.0002] |
| FewShot-R18 jogadores (congelado) | `frozen_ctx4_pos_minus_neg_k1` | 0.5112 | 0.7406 | 0.7553 | 0.6805 | 0.7160 | 0.6028 | 8/742 | 2/2268 | +0.0005 [-0.0006, +0.0018] |
| FewShot-R18 jogadores (congelado) | `frozen_ctx4_pos_minus_neg_k6` | 0.5386 | 0.7708 | 0.7537 | 0.6811 | 0.7156 | 0.6032 | 1/742 | 0/2268 | +0.0001 [+0.0000, +0.0005] |
| FewShot-R18 jogadores (congelado) | `frozen_ctx4_positive_only_k1` | 0.4515 | 0.7115 | 0.7535 | 0.6811 | 0.7155 | 0.6032 | 0/742 | 0/2268 | +0.0000 [+0.0000, +0.0000] |
| FewShot-R18 jogadores (congelado) | `frozen_ctx4_positive_only_k6` | 0.5211 | 0.7516 | 0.7535 | 0.6802 | 0.7150 | 0.6024 | 1/742 | 3/2268 | -0.0005 [-0.0018, +0.0001] |
| FewShot-R18-bola (treinado layer3/4) | `trained_ball_layer34_ctx2_pos_minus_neg_k1` | 0.6953 | 0.8386 | 0.7852 | 0.6706 | 0.7234 | 0.6014 | 131/742 | 35/2268 | +0.0079 [+0.0028, +0.0141] |
| FewShot-R18-bola (treinado layer3/4) | `trained_ball_layer34_ctx2_pos_minus_neg_k6` | 0.6948 | 0.8395 | 0.7850 | 0.6709 | 0.7234 | 0.6015 | 130/742 | 34/2268 | +0.0080 [+0.0030, +0.0145] |
| FewShot-R18-bola (treinado layer3/4) | `trained_ball_layer34_ctx2_positive_only_k1` | 0.6948 | 0.8390 | 0.7869 | 0.6709 | 0.7243 | 0.6018 | 137/742 | 34/2268 | +0.0088 [+0.0038, +0.0151] |
| FewShot-R18-bola (treinado layer3/4) | `trained_ball_layer34_ctx2_positive_only_k6` | 0.6341 | 0.7932 | 0.7855 | 0.6709 | 0.7237 | 0.6017 | 132/742 | 34/2268 | +0.0082 [+0.0032, +0.0145] |
| FewShot-R18-bola (treinado layer3/4) | `trained_ball_layer34_ctx3_pos_minus_neg_k1` | 0.7131 | 0.8672 | 0.7750 | 0.6724 | 0.7201 | 0.6010 | 91/742 | 30/2268 | +0.0046 [-0.0004, +0.0093] |
| FewShot-R18-bola (treinado layer3/4) | `trained_ball_layer34_ctx3_pos_minus_neg_k6` | 0.7128 | 0.8652 | 0.7750 | 0.6724 | 0.7201 | 0.6010 | 91/742 | 30/2268 | +0.0046 [-0.0004, +0.0093] |
| FewShot-R18-bola (treinado layer3/4) | `trained_ball_layer34_ctx3_positive_only_k1` | 0.7084 | 0.8630 | 0.7787 | 0.6709 | 0.7208 | 0.6007 | 106/742 | 35/2268 | +0.0053 [-0.0002, +0.0103] |
| FewShot-R18-bola (treinado layer3/4) | `trained_ball_layer34_ctx3_positive_only_k6` | 0.5297 | 0.7507 | 0.7787 | 0.6709 | 0.7208 | 0.6007 | 106/742 | 35/2268 | +0.0053 [-0.0002, +0.0103] |
| FewShot-R18-bola (treinado layer3/4) | `trained_ball_layer34_ctx4_pos_minus_neg_k1` | 0.6994 | 0.8688 | 0.7815 | 0.6628 | 0.7173 | 0.5946 | 121/742 | 65/2268 | +0.0018 [-0.0069, +0.0059] |
| FewShot-R18-bola (treinado layer3/4) | `trained_ball_layer34_ctx4_pos_minus_neg_k6` | 0.6922 | 0.8649 | 0.7818 | 0.6628 | 0.7174 | 0.5946 | 122/742 | 65/2268 | +0.0019 [-0.0067, +0.0061] |
| FewShot-R18-bola (treinado layer3/4) | `trained_ball_layer34_ctx4_positive_only_k1` | 0.6996 | 0.8671 | 0.7815 | 0.6628 | 0.7173 | 0.5946 | 121/742 | 65/2268 | +0.0018 [-0.0069, +0.0059] |
| FewShot-R18-bola (treinado layer3/4) | `trained_ball_layer34_ctx4_positive_only_k6` | 0.5742 | 0.7824 | 0.7760 | 0.6658 | 0.7167 | 0.5961 | 100/742 | 53/2268 | +0.0012 [-0.0064, +0.0051] |
| FewShot-R18-bola (treinado todas camadas) | `trained_ball_all_layers_ctx2_pos_minus_neg_k1` | 0.6116 | 0.8019 | 0.7699 | 0.6724 | 0.7179 | 0.6021 | 72/742 | 30/2268 | +0.0024 [-0.0027, +0.0062] |
| FewShot-R18-bola (treinado todas camadas) | `trained_ball_all_layers_ctx2_pos_minus_neg_k6` | 0.6159 | 0.7979 | 0.7694 | 0.6703 | 0.7164 | 0.5995 | 72/742 | 37/2268 | +0.0010 [-0.0035, +0.0053] |
| FewShot-R18-bola (treinado todas camadas) | `trained_ball_all_layers_ctx2_positive_only_k1` | 0.6011 | 0.7910 | 0.7698 | 0.6718 | 0.7174 | 0.6012 | 72/742 | 32/2268 | +0.0020 [-0.0033, +0.0056] |
| FewShot-R18-bola (treinado todas camadas) | `trained_ball_all_layers_ctx2_positive_only_k6` | 0.6196 | 0.8074 | 0.7698 | 0.6718 | 0.7174 | 0.6012 | 72/742 | 32/2268 | +0.0020 [-0.0033, +0.0056] |
| FewShot-R18-bola (treinado todas camadas) | `trained_ball_all_layers_ctx3_pos_minus_neg_k1` | 0.5848 | 0.7754 | 0.7657 | 0.6754 | 0.7177 | 0.6021 | 53/742 | 20/2268 | +0.0023 [-0.0007, +0.0058] |
| FewShot-R18-bola (treinado todas camadas) | `trained_ball_all_layers_ctx3_pos_minus_neg_k6` | 0.6057 | 0.7925 | 0.7673 | 0.6763 | 0.7189 | 0.6021 | 59/742 | 16/2268 | +0.0035 [+0.0002, +0.0078] |
| FewShot-R18-bola (treinado todas camadas) | `trained_ball_all_layers_ctx3_positive_only_k1` | 0.6410 | 0.8121 | 0.7652 | 0.6745 | 0.7170 | 0.6008 | 52/742 | 23/2268 | +0.0015 [-0.0026, +0.0052] |
| FewShot-R18-bola (treinado todas camadas) | `trained_ball_all_layers_ctx3_positive_only_k6` | 0.6507 | 0.8215 | 0.7646 | 0.6739 | 0.7164 | 0.5995 | 51/742 | 24/2268 | +0.0009 [-0.0027, +0.0054] |
| FewShot-R18-bola (treinado todas camadas) | `trained_ball_all_layers_ctx4_pos_minus_neg_k1` | 0.5333 | 0.7561 | 0.7561 | 0.6787 | 0.7153 | 0.6015 | 12/742 | 9/2268 | -0.0002 [-0.0015, +0.0026] |
| FewShot-R18-bola (treinado todas camadas) | `trained_ball_all_layers_ctx4_pos_minus_neg_k6` | 0.6231 | 0.8149 | 0.7769 | 0.6682 | 0.7184 | 0.5991 | 103/742 | 43/2268 | +0.0030 [-0.0041, +0.0075] |
| FewShot-R18-bola (treinado todas camadas) | `trained_ball_all_layers_ctx4_positive_only_k1` | 0.5919 | 0.7894 | 0.7767 | 0.6685 | 0.7185 | 0.5992 | 102/742 | 42/2268 | +0.0031 [-0.0039, +0.0076] |
| FewShot-R18-bola (treinado todas camadas) | `trained_ball_all_layers_ctx4_positive_only_k6` | 0.5856 | 0.7846 | 0.7767 | 0.6685 | 0.7185 | 0.5992 | 102/742 | 42/2268 | +0.0031 [-0.0039, +0.0076] |

### Mineração no `train.txt` inteiro e treino episódico (Rodada 2)

- Mineração completa: `32852` imagens de `train.txt` varridas (`imgsz=1280`, `conf>=0.05`): `26736` TPs e `1820` FPs em `conf>=0.35` (`0.0554` FP/img, precisão `0.9363`) + `21172` FPs minerados em baixa confiança (`0.05..0.35`), `0` recortes artificiais de fundo.
- Comparação de taxa de FP (viés de treino do detector): `train.txt` = `0.0554` FP/img (a `imgsz=1280`) vs `val.txt` = `0.2285` FP/img (`742` FPs em `3247` imagens a `imgsz=1920`).
- Receita de treino: ResNet-18 ImageNet (`fc=Identity`, `BatchNorm2d` em `eval`), recorte quadrado `96x96`, `AdamW(lr=2e-4, wd=1e-4)`, `CosineAnnealingLR`, `16` épocas × `360` episódios = `5760` episódios (4× a Rodada 1), `cosine * 20`, seed `25`. Testado para `layer3/4` (receita do paper) e variação `all_layers` (todas as camadas treináveis), nos contextos `2×`, `3×` e `4×`.

| Braço | Contexto | Camadas | Pool pos / neg | Loss época 1 | Loss época final | Tempo treino (s) |
|---|---:|---|---|---:|---:|---:|
| FewShot-R18-bola (treinado layer3/4) | 2× | `layer3_layer4` | 12000 / 8329 | 0.5988 | 0.1025 | 275.1 |
| FewShot-R18-bola (treinado todas camadas) | 2× | `all` | 12000 / 8329 | 0.6770 | 0.6931 | 368.2 |
| FewShot-R18-bola (treinado layer3/4) | 3× | `layer3_layer4` | 12000 / 8329 | 0.5680 | 0.0638 | 224.9 |
| FewShot-R18-bola (treinado todas camadas) | 3× | `all` | 12000 / 8329 | 0.6680 | 0.6931 | 319.9 |
| FewShot-R18-bola (treinado layer3/4) | 4× | `layer3_layer4` | 12000 / 8329 | 0.5693 | 0.0554 | 220.8 |
| FewShot-R18-bola (treinado todas camadas) | 4× | `all` | 12000 / 8329 | 0.6838 | 0.6931 | 311.5 |

## Auditoria de rótulo ruidoso no `val.txt` (`results/amostra_fp.png`)

Dos `393` FPs do detector em `conf >= 0.5` no `val.txt`, sorteamos uma amostra fixa de `40` recortes (seed `42`, margem `3×`, caixa vermelha = detecção YOLO), salva em `results/amostra_fp.png` e inspecionada visualmente:

- **Bolas reais não anotadas (bolas extras fora de jogo / linha / gol)**: `9/40` (22.5%, IC Wilson 95%: [12.3%, 37.5%]). Índices: `[2, 5, 11, 23, 24, 25, 28, 29, 31]`.
- **Bola real já anotada com IoU < 0,50 ou caixa duplicada**: `22/40` (55.0%). Índices: `[1, 3, 4, 7, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 26, 27, 30, 33, 34, 35, 36, 40]`.
- **Duvidosos** (pequenos/borrados, provável bola extra): `2/40` (5.0%). Índices: `[6, 37]`.
- **Falsos positivos genuínos não-bola** (letras 'O'/'2' em placas, ícone de bola no placar de TV, cabeças): `7/40` (17.5%). Índices: `[8, 9, 17, 22, 32, 38, 39]`.
- Quebra programática de TODOS os `742` FPs em `val.txt` (`conf >= 0.35`): `33` duplicados (`IoU >= 0.50`), `248` quase-acerto (`0.30 <= IoU < 0.50`), `37` sobreposição parcial (`0.10 <= IoU < 0.30`), e `424` distantes do GT (`IoU < 0.10`).
- Notas da inspeção visual: 20/40 são a própria bola anotada com pequeno deslocamento em caixas minúsculas de 5–12 px (0,34<=IoU<0,50, dist<=4,4 px) e 2/40 (#07, #10) são caixas duplicadas sobre a mesma bola anotada (IoU>=0,55). 9/40 são bolas extras reais paradas no gol/linha lateral não anotadas (#02,#05,#11,#23,#24,#25,#28,#29,#31) e 2/40 (#06,#37) são prováveis bolas extras borradas. Apenas 7/40 (17,5%) são FPs genuínos não-bola: ícone de bola no placar de TV (#08), letras 'O'/'2' em placas (#09,#17,#22,#32) e cabeças de goleiro/árbitro (#38,#39).

**Estimativa de métricas descontando bolas reais não anotadas** (estimativa conservadora baseada na amostra de 40 recortes, assumindo a mesma taxa entre FPs mantidos e removidos; **nenhum rótulo original foi alterado**):

| Cenário (estimativa) | Precisão base | Recall base | F1 base | Precisão filtrada | Recall filtrado | F1 filtrado | ΔF1 estimado |
|---|---:|---:|---:|---:|---:|---:|---:|
| Taxa aplicada a todos os FPs (22.5%) | 0.8090 | 0.6963 | 0.7484 | 0.8323 | 0.6769 | 0.7466 | -0.0019 |
| Taxa aplicada apenas aos FPs `conf>=0.50` (393/742) | 0.7827 | 0.6893 | 0.7330 | 0.8094 | 0.6735 | 0.7352 | +0.0022 |

## Procedência e auditoria do `soccernet_gsr_test_virgin` (descartado como prova)

- **Origem**: `/Users/j.henrique/soccer-four-ref-gsr25/archives/test.zip` (split `test` do SoccerNet Game State Reconstruction 2025, 49 clipes 1080p de transmissão).
- **Montagem na Rodada 1**: `10 quadros igualmente espaçados por clipe em 49 clipes do split test do SoccerNet GSR 2025 (490 quadros, 456 anotações category_id=4 role=ball).`.
- **Prova de 0 sobreposição com `r1`**: `0 correspondências de SHA-256 contra todas as 36.150 imagens listadas em r1/copy_receipt.json (e fora dos jogos de treino 4, 6, 9 do FewShot-R18).`.
- **Confiabilidade das anotações de bola (auditoria visual)**: NÃO CONFIÁVEL — descartado como prova na Rodada 2. A auditoria visual de 20 caixas GT e 8 'FPs' do t1b_v5 no GSR mostrou que as anotações de bola do SoccerNet GSR (dataset voltado a jogadores/estado de jogo) ficam frequentemente deslocadas/atrasadas em bolas rápidas (inclusive caixas GT sobre grama vazia). Como consequência, 6–7 de 8 'FPs' inspecionados eram bolas reais detectadas corretamente pelo YOLO com IoU < 0,50 contra a caixa deslocada do GSR, corrompendo artificialmente precisão, F1 e AUC.
- **`box2boxcam` (câmera amadora/noturna, 198 quadros)**: não disponível localmente; permanece não medido.

## Rodada 1 (preservada como referência histórica)

Arquivo completo preservado em `results/rodada1_experiment_results.json`. Na Rodada 1, o protocolo usou recorte justo `context_scale=1.0` (`48x96` letterbox), partição única 50/50 de `val.txt` (`val_calib` 1.624 imgs / 642 jogos, F1 base `0.6381`; `val_eval` 1.623 imgs / 151 jogos contendo `v14`, F1 base `0.7887`), mineração em 1.500 imagens de `train.txt` e 8 épocas × 180 episódios (1.440 episódios), com `12` configurações testadas:

| Braço (Rodada 1, `val_eval`) | Config selecionada | Precisão | Recall | F1 | mAP50 | ROC-AUC | FPs rem | TPs rem | ΔF1 (IC 95% imgs) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Detector sozinho | baseline | 0.7960 | 0.7815 | 0.7887 | 0.7251 | 0.7786 (`conf`) | 0/329 | 0/1284 | 0.0000 |
| FewShot-R18 jogadores (congelado) | `layer2_positive_only_k1` | 0.7984 | 0.7785 | 0.7883 | 0.7218 | 0.5125 | 6/329 | 5/1284 | -0.0004 [-0.0024, +0.0013] |
| FewShot-R18-bola (treinado) | `trained_ball_layer4_pos_minus_neg_k6` | 0.8278 | 0.7139 | 0.7667 | 0.6685 | 0.6244 | 85/329 | 111/1284 | -0.0220 [-0.0309, -0.0129] |

## Limitações

- `val.txt` não é teste virgem do detector: foi usado para selecionar o checkpoint `t1b_v5_etapa1_rodada1.pt`. Os valores absolutos podem ser otimistas, embora a comparação entre os braços ocorra sob o mesmo viés.
- O grupo `v14` concentra 1.111 das 3.247 imagens de `val.txt` (34,2%) sob um único `game_id`, aumentando a variância do bootstrap por jogo e criando uma dobra maior na CV de 5 dobras.
- O filtro só remove detecções (nunca gera novas caixas nem aumenta recall); quando o detector já tem precisão próxima de 0,75–0,80, qualquer remoção imprecisa custa caro em F1.
- Parte dos FPs registrados em `val.txt` corresponde a bolas reais não anotadas (auditado em `results/amostra_fp.png`), o que penaliza tanto o detector quanto qualquer filtro que acerte ao manter essas bolas.
- A mineração completa em `train.txt` (32.852 imagens) usou `imgsz=1280` (resolução de treino do `t1b_v5`) por viabilidade computacional no Mac MPS, enquanto a avaliação oficial em `val.txt` manteve `imgsz=1920` (padrão do site).
- Não havia conjunto virgem confiável de câmera amadora/noturna (`box2boxcam`, 198 quadros) disponível localmente; `soccernet_gsr_test_virgin` foi auditado visualmente e descartado por caixas GT de bola deslocadas.
