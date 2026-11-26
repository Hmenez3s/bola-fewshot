from __future__ import annotations

from typing import Any


def _fmt(value: float | None, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def _fmt_signed(value: float | None, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    return f"{value:+.{digits}f}"


def _fmt_threshold(value: Any) -> str:
    if value == "-inf":
        return "-inf (sem corte)"
    if value == "inf":
        return "+inf"
    if isinstance(value, (int, float)):
        return f"{value:.4f}"
    return str(value)


def _dir_pt(direction: str) -> str:
    if direction == "improvement":
        return "melhora distinguível de zero"
    if direction == "degradation":
        return "piora distinguível de zero"
    return "indistinguível de zero"


def _dir_short(direction: str) -> str:
    if direction == "improvement":
        return "melhora"
    if direction == "degradation":
        return "piora"
    return "indistinguível"


def _arm_label(arm: str) -> str:
    return {
        "frozen_players": "FewShot-R18 jogadores (congelado)",
        "trained_ball_layer34": "FewShot-R18-bola (treinado layer3/4)",
        "trained_ball_all_layers": "FewShot-R18-bola (treinado todas camadas)",
    }.get(arm, arm)


def render_combined_report(
    round1: dict[str, Any],
    round2: dict[str, Any],
    fp_audit_report: dict[str, Any],
    gsr_audit_note: dict[str, Any],
) -> str:
    p2 = round2["protocol"]
    s2 = round2["splits"]
    base2 = round2["baseline_val_all"]
    det_sep2 = round2["detector_conf_separability"]
    nested = round2["nested_arms"]
    m_stats = round2["mining_stats"]

    arm_frz = nested["frozen_players"]
    arm_l34 = nested["trained_ball_layer34"]
    arm_all = nested.get("trained_ball_all_layers")

    best_trained = arm_l34
    if arm_all is not None and arm_all["pooled_oof_metrics"]["f1"] > arm_l34["pooled_oof_metrics"]["f1"]:
        best_trained = arm_all

    bt_m = best_trained["pooled_oof_metrics"]
    bt_c = best_trained["pooled_oof_cost"]
    bt_bi = best_trained["bootstrap_image"]["delta_f1"]
    bt_bg = best_trained["bootstrap_group"]["delta_f1"]
    frz_m = arm_frz["pooled_oof_metrics"]
    frz_bg = arm_frz["bootstrap_group"]["delta_f1"]
    l34_m = arm_l34["pooled_oof_metrics"]
    l34_bg = arm_l34["bootstrap_group"]["delta_f1"]

    r1_base = round1["baseline_eval"]
    r1_tr_id = round1["selected_trained_config_id"]
    r1_tr = next(c for c in round1["configurations"] if c["config_id"] == r1_tr_id)
    r1_tr_df1 = r1_tr["bootstrap"]["delta_f1"]["point_estimate"]

    changed = bt_bg["direction"] == "improvement" or bt_bi["direction"] == "improvement"
    change_verdict = (
        f"A conclusão MUDOU na Rodada 2: com contexto ampliado, CV de 5 dobras por jogo e treino completo em train.txt, {_arm_label(best_trained['arm'])} obteve {_dir_pt(bt_bg['direction'])} (ΔF1={_fmt_signed(bt_bg['point_estimate'])}, contra ΔF1={_fmt_signed(r1_tr_df1)} na Rodada 1)."
        if changed
        else f"A conclusão NÃO mudou na Rodada 2: mesmo corrigindo contexto (2×/3×/4×), CV de 5 dobras por jogo, mineração em todo o train.txt ({m_stats['train_images_scanned']} imagens) e 4× mais episódios ({p2['train_total_episodes']}), o filtro não supera o detector sozinho (melhor braço treinado {_arm_label(best_trained['arm'])}: ΔF1={_fmt_signed(bt_bg['point_estimate'])}, {_dir_pt(bt_bg['direction'])}; Rodada 1: ΔF1={_fmt_signed(r1_tr_df1)})."
    )

    chosen_ctxs_l34 = [f"f{f['fold']}:{f['chosen_config_id']}" for f in arm_l34["folds"]]
    fp_est = fp_audit_report["noise_adjustment"]
    fp_strict = fp_est["adjusted_applying_rate_to_all_fps"]

    all_layers_part = ""
    if arm_all is not None:
        am = arm_all["pooled_oof_metrics"]
        all_layers_part = f" | Treinado todas camadas P={_fmt(am['precision'])}/R={_fmt(am['recall'])}/F1={_fmt(am['f1'])}/mAP50={_fmt(am['map50'])}"

    summary_lines = [
        f"1. Veredito (Rodada 1 vs Rodada 2): {change_verdict}",
        f"2. Correções aplicadas na Rodada 2: recorte quadrado `{p2['canvas_wh'][0]}x{p2['canvas_wh'][1]}` com `context_scale` ∈ `{p2['context_scales']}` escolhido só na calibração, CV aninhada de {p2['n_folds']} dobras por jogo sobre todo o `val.txt` ({s2['val_images']} imagens, {s2['train_val_group_audit']['groups_b']} jogos; `train∩val=0`), mineração em {m_stats['train_images_scanned']} imagens de `train.txt` e {p2['train_total_episodes']} episódios.",
        f"3. Métricas OOF na Rodada 2 (todo `val.txt`, imgsz={p2['imgsz']}, conf={p2['conf']}): Base P={_fmt(base2['precision'])}/R={_fmt(base2['recall'])}/F1={_fmt(base2['f1'])}/mAP50={_fmt(base2['map50'])} | Congelado P={_fmt(frz_m['precision'])}/R={_fmt(frz_m['recall'])}/F1={_fmt(frz_m['f1'])}/mAP50={_fmt(frz_m['map50'])} | Treinado layer3/4 P={_fmt(l34_m['precision'])}/R={_fmt(l34_m['recall'])}/F1={_fmt(l34_m['f1'])}/mAP50={_fmt(l34_m['map50'])}{all_layers_part}.",
        f"4. Custo real ({_arm_label(best_trained['arm'])}): removeu {bt_c['fps_removed']}/{bt_c['base_fp']} FPs ({_fmt(bt_c['fp_removal_rate']*100, 1)}%) e removeu {bt_c['tps_removed']}/{bt_c['base_tp']} bolas verdadeiras por engano (ΔRecall={_fmt_signed(bt_m['recall'] - base2['recall'])}).",
        f"5. Bootstrap pareado ({p2['bootstrap_resamples']} reamostragens) para `{best_trained['arm']}` vs Base: jogos (n={best_trained['bootstrap_group']['n_groups']}) ΔF1={_fmt_signed(bt_bg['point_estimate'])} IC 95% [{_fmt_signed(bt_bg['ci_95_low'])}, {_fmt_signed(bt_bg['ci_95_high'])}] ({_dir_short(bt_bg['direction'])}); imagens (n={s2['val_images']}) IC 95% [{_fmt_signed(bt_bi['ci_95_low'])}, {_fmt_signed(bt_bi['ci_95_high'])}]; média das 5 dobras={_fmt_signed(best_trained['mean_fold_delta_f1'])}±{_fmt(best_trained['sd_fold_delta_f1'])}.",
        f"6. Separabilidade OOF bola vs FP (ROC-AUC): `conf` do detector YOLO={_fmt(det_sep2['roc_auc'])}; jogadores congelado={_fmt(arm_frz['pooled_oof_roc_auc'])}; bola treinado layer3/4={_fmt(arm_l34['pooled_oof_roc_auc'])}"
        + (f"; bola treinado todas camadas={_fmt(arm_all['pooled_oof_roc_auc'])}." if arm_all is not None else "."),
        f"7. Rótulo ruidoso (`results/amostra_fp.png`, {fp_est['n_audited']} FPs em conf>=0.50): {fp_est['n_unannotated_real_balls']}/{fp_est['n_audited']} ({_fmt(fp_est['unannotated_ball_rate_strict']*100, 1)}%, IC Wilson [{_fmt(fp_est['unannotated_ball_rate_wilson_95'][0]*100, 1)}%, {_fmt(fp_est['unannotated_ball_rate_wilson_95'][1]*100, 1)}%]) são bolas reais não anotadas (+{fp_est['n_ambiguous']} duvidosos); descontando essa estimativa, F1 base={_fmt(fp_strict['baseline']['f1_adjusted'])} vs filtrado={_fmt(fp_strict['filtered']['f1_adjusted'])} (ΔF1 estimado={_fmt_signed(fp_strict['delta_f1_adjusted'])}). `soccernet_gsr_test_virgin` foi descartado como prova (anotações GT de bola deslocadas/atrasadas).",
        f"8. Viés e teto de ganho: `val.txt` não é teste virgem do detector (usado na escolha do checkpoint; comparação relativa justa, absolutos possivelmente otimistas); FP/img `train`={m_stats['train_fp_per_image_at_eval_conf']:.4f} vs `val`={base2['fp_per_image']:.4f}; headroom base em `val`=+0.0000→{_fmt_signed(base2['headroom']['precision_headroom'])} em precisão e {_fmt_signed(base2['headroom']['f1_headroom'])} em F1; configurações testadas: Rodada 1={round1['protocol']['total_configs_tested']}, Rodada 2={p2['total_fixed_configs_tested']} fixas + {p2['total_nested_arm_selectors']} seletores CV aninhados.",
    ]

    lines: list[str] = [
        "# Relatório: Detector de bola + filtro FewShot-R18 (Rodada 1 e Rodada 2)",
        "",
        "## Resumo (8 linhas)",
        "",
        *summary_lines,
        "",
        "## Rodada 2 — Validação cruzada aninhada (5 dobras por jogo, contexto 2×/3×/4×, treino completo)",
        "",
        (
            f"Detector: `{p2['detector_weights']}` (`imgsz={p2['imgsz']}`, `conf={p2['conf']}`, `IoU={p2['eval_iou']}`) | "
            f"Conjunto avaliado: todo o `val.txt` ({s2['val_images']} imagens, {s2['val_gt_balls']} bolas GT, "
            f"{s2['train_val_group_audit']['groups_b']} jogos; `train∩val={s2['train_val_group_audit']['shared_groups']}` jogos). "
            f"Aviso de viés: `val.txt` foi usado pelo detector para escolher o melhor checkpoint (não é teste virgem; comparação entre braços sob o mesmo viés)."
        ),
        "",
        "### Comparação OOF agregada dos braços (seleção aninhada de contexto, K, época e limiar por dobra)",
        "",
        "| Braço | Precisão | Recall | F1 | mAP50 | mAP50-95 | FPs removidos | TPs removidos | ROC-AUC OOF | ΔF1 (IC 95% jogos) | ΔF1 (IC 95% imgs) | Média±DP (5 dobras) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        (
            f"| Detector sozinho (baseline) | {_fmt(base2['precision'])} | {_fmt(base2['recall'])} | "
            f"{_fmt(base2['f1'])} | {_fmt(base2['map50'])} | {_fmt(base2['map50_95'])} | "
            f"0 / {base2['fp']} | 0 / {base2['tp']} | {_fmt(det_sep2['roc_auc'])} (`conf`) | 0.0000 | 0.0000 | 0.0000 |"
        ),
    ]

    for arm_key in ("frozen_players", "trained_ball_layer34", "trained_ball_all_layers"):
        if arm_key not in nested:
            continue
        ar = nested[arm_key]
        am = ar["pooled_oof_metrics"]
        ac = ar["pooled_oof_cost"]
        bg = ar["bootstrap_group"]["delta_f1"]
        bi = ar["bootstrap_image"]["delta_f1"]
        lines.append(
            f"| {_arm_label(arm_key)} | {_fmt(am['precision'])} | {_fmt(am['recall'])} | {_fmt(am['f1'])} | "
            f"{_fmt(am['map50'])} | {_fmt(am['map50_95'])} | {ac['fps_removed']} / {ac['base_fp']} | "
            f"{ac['tps_removed']} / {ac['base_tp']} | {_fmt(ar['pooled_oof_roc_auc'])} | "
            f"{_fmt_signed(bg['point_estimate'])} [{_fmt_signed(bg['ci_95_low'])}, {_fmt_signed(bg['ci_95_high'])}] | "
            f"[{_fmt_signed(bi['ci_95_low'])}, {_fmt_signed(bi['ci_95_high'])}] | "
            f"{_fmt_signed(ar['mean_fold_delta_f1'])} ± {_fmt(ar['sd_fold_delta_f1'])} |"
        )

    lines.extend(["", "### Escolhas congeladas por dobra (sem olhar a dobra de teste)", ""])
    for arm_key in ("frozen_players", "trained_ball_layer34", "trained_ball_all_layers"):
        if arm_key not in nested:
            continue
        ar = nested[arm_key]
        lines.extend(
            [
                f"#### {_arm_label(arm_key)}",
                "",
                "| Dobra | Imagens | Config escolhida (calib) | Época | Limiar congelado | AUC calib | AUC dobra | F1 base dobra | F1 filt dobra | ΔF1 dobra | FPs rem | TPs rem |",
                "|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for fl in ar["folds"]:
            lines.append(
                f"| {fl['fold']} | {fl['n_images']} | `{fl['chosen_config_id']}` | {fl['chosen_epoch']} | "
                f"{_fmt_threshold(fl['frozen_threshold'])} | {_fmt(fl['calib_roc_auc'])} | {_fmt(fl['fold_roc_auc'])} | "
                f"{_fmt(fl['fold_base_f1'])} | {_fmt(fl['fold_filt_f1'])} | {_fmt_signed(fl['fold_delta_f1'])} | "
                f"{fl['fps_removed']} | {fl['tps_removed']} |"
            )
        bd = ar.get("v14_breakdown")
        if bd:
            lines.append(
                f"- Decomposição `v14` vs resto: `v14` ({bd['v14_only']['n_images']} imgs) F1 base={_fmt(bd['v14_only']['base_f1'])} → filt={_fmt(bd['v14_only']['filt_f1'])} (Δ={_fmt_signed(bd['v14_only']['delta_f1'])}); "
                f"demais 792 jogos ({bd['non_v14_792_games']['n_images']} imgs) F1 base={_fmt(bd['non_v14_792_games']['base_f1'])} → filt={_fmt(bd['non_v14_792_games']['filt_f1'])} (Δ={_fmt_signed(bd['non_v14_792_games']['delta_f1'])})."
            )
        lines.append("")

    lines.extend(
        [
            "### Comparação direta entre braços treinados e controle congelado (bootstrap pareado OOF)",
            "",
        ]
    )
    for comp_key, comp_val in round2.get("arm_comparisons", {}).items():
        cg = comp_val["bootstrap_group"]["delta_f1"]
        ci = comp_val["bootstrap_image"]["delta_f1"]
        lines.append(
            f"- `{comp_key}`: ΔF1={_fmt_signed(cg['point_estimate'])} | "
            f"IC 95% jogos [{_fmt_signed(cg['ci_95_low'])}, {_fmt_signed(cg['ci_95_high'])}] ({_dir_short(cg['direction'])}) | "
            f"IC 95% imagens [{_fmt_signed(ci['ci_95_low'])}, {_fmt_signed(ci['ci_95_high'])}] ({_dir_short(ci['direction'])})."
        )

    lines.extend(
        [
            "",
            "### Todas as configurações fixas testadas na Rodada 2 (OOF 5 dobras)",
            "",
            "| Braço | Configuração | ROC-AUC OOF | PR-AUC OOF | Precisão | Recall | F1 | mAP50 | FPs rem | TPs rem | ΔF1 (IC 95% jogos) |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for cfg in round2["fixed_configs"]:
        cm = cfg["pooled_oof_metrics"]
        cc = cfg["pooled_oof_cost"]
        cbg = cfg["bootstrap_group"]["delta_f1"]
        lines.append(
            f"| {_arm_label(cfg['arm'])} | `{cfg['config_id']}` | {_fmt(cfg['pooled_oof_roc_auc'])} | "
            f"{_fmt(cfg['pooled_oof_pr_auc'])} | {_fmt(cm['precision'])} | {_fmt(cm['recall'])} | "
            f"{_fmt(cm['f1'])} | {_fmt(cm['map50'])} | {cc['fps_removed']}/{cc['base_fp']} | "
            f"{cc['tps_removed']}/{cc['base_tp']} | {_fmt_signed(cbg['point_estimate'])} "
            f"[{_fmt_signed(cbg['ci_95_low'])}, {_fmt_signed(cbg['ci_95_high'])}] |"
        )

    lines.extend(
        [
            "",
            "### Mineração no `train.txt` inteiro e treino episódico (Rodada 2)",
            "",
            f"- Mineração completa: `{m_stats['train_images_scanned']}` imagens de `train.txt` varridas (`imgsz={p2['train_mining_imgsz']}`, `conf>={m_stats['mining_conf']}`): `{m_stats['train_tp_at_eval_conf']}` TPs e `{m_stats['train_fp_at_eval_conf']}` FPs em `conf>={m_stats['eval_conf']}` (`{m_stats['train_fp_per_image_at_eval_conf']:.4f}` FP/img, precisão `{m_stats['train_precision_at_eval_conf']:.4f}`) + `{m_stats['train_fp_low_conf_mined']}` FPs minerados em baixa confiança (`{m_stats['mining_conf']}..{m_stats['eval_conf']}`), `0` recortes artificiais de fundo.",
            f"- Comparação de taxa de FP (viés de treino do detector): `train.txt` = `{m_stats['train_fp_per_image_at_eval_conf']:.4f}` FP/img (a `imgsz={p2['train_mining_imgsz']}`) vs `val.txt` = `{base2['fp_per_image']:.4f}` FP/img (`{base2['fp']}` FPs em `{base2['n_images']}` imagens a `imgsz={p2['imgsz']}`).",
            f"- Receita de treino: ResNet-18 ImageNet (`fc=Identity`, `BatchNorm2d` em `eval`), recorte quadrado `{p2['canvas_wh'][0]}x{p2['canvas_wh'][1]}`, `AdamW(lr=2e-4, wd=1e-4)`, `CosineAnnealingLR`, `{p2['train_epochs']}` épocas × `{p2['train_steps_per_epoch']}` episódios = `{p2['train_total_episodes']}` episódios (4× a Rodada 1), `cosine * 20`, seed `{p2['train_seed']}`. Testado para `layer3/4` (receita do paper) e variação `all_layers` (todas as camadas treináveis), nos contextos `2×`, `3×` e `4×`.",
            "",
            "| Braço | Contexto | Camadas | Pool pos / neg | Loss época 1 | Loss época final | Tempo treino (s) |",
            "|---|---:|---|---|---:|---:|---:|",
        ]
    )
    for tr in round2["training_runs"]:
        hist = tr["history"]
        lines.append(
            f"| {_arm_label(tr['arm'])} | {tr['context_scale']:.0f}× | `{tr['trainable_layers']}` | "
            f"{tr['pool_stats']['pool_pos_crops']} / {tr['pool_stats']['pool_neg_crops']} | "
            f"{hist[0]['loss']:.4f} | {hist[-1]['loss']:.4f} | {tr['total_train_s']:.1f} |"
        )

    lines.extend(
        [
            "",
            "## Auditoria de rótulo ruidoso no `val.txt` (`results/amostra_fp.png`)",
            "",
            (
                f"Dos `{fp_audit_report['grid_meta']['total_fps_at_min_conf']}` FPs do detector em `conf >= {fp_audit_report['grid_meta']['min_conf']}` "
                f"no `val.txt`, sorteamos uma amostra fixa de `{fp_est['n_audited']}` recortes (seed `{fp_audit_report['grid_meta']['seed']}`, "
                f"margem `3×`, caixa vermelha = detecção YOLO), salva em `results/amostra_fp.png` e inspecionada visualmente:"
            ),
            "",
            f"- **Bolas reais não anotadas**: `{fp_est['n_unannotated_real_balls']}/{fp_est['n_audited']}` ({_fmt(fp_est['unannotated_ball_rate_strict']*100, 1)}%, IC Wilson 95%: [{_fmt(fp_est['unannotated_ball_rate_wilson_95'][0]*100, 1)}%, {_fmt(fp_est['unannotated_ball_rate_wilson_95'][1]*100, 1)}%]). Índices: `{fp_audit_report['unannotated_indices']}`.",
            f"- **Duvidosos** (pequenos/borrados): `{fp_est['n_ambiguous']}/{fp_est['n_audited']}` ({_fmt((fp_est['n_ambiguous']/max(1, fp_est['n_audited']))*100, 1)}%). Índices: `{fp_audit_report['ambiguous_indices']}`.",
            f"- **Falsos positivos reais** (chuteira, meia, cabeça, linha, placa, torcida, luz, etc.): `{fp_est['n_true_fps']}/{fp_est['n_audited']}` ({_fmt((fp_est['n_true_fps']/max(1, fp_est['n_audited']))*100, 1)}%).",
            f"- Notas da inspeção visual: {fp_audit_report['visual_notes']}",
            "",
            "**Estimativa de métricas descontando bolas reais não anotadas** (estimativa baseada na amostra de 40 recortes; **nenhum rótulo original foi alterado**):",
            "",
            "| Cenário (estimativa) | Precisão base | Recall base | F1 base | Precisão filtrada | Recall filtrado | F1 filtrado | ΔF1 estimado |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
            (
                f"| Taxa aplicada a todos os FPs ({_fmt(fp_est['unannotated_ball_rate_strict']*100, 1)}%) | "
                f"{_fmt(fp_strict['baseline']['precision_adjusted'])} | {_fmt(fp_strict['baseline']['recall_adjusted'])} | "
                f"{_fmt(fp_strict['baseline']['f1_adjusted'])} | {_fmt(fp_strict['filtered']['precision_adjusted'])} | "
                f"{_fmt(fp_strict['filtered']['recall_adjusted'])} | {_fmt(fp_strict['filtered']['f1_adjusted'])} | "
                f"{_fmt_signed(fp_strict['delta_f1_adjusted'])} |"
            ),
            (
                f"| Taxa aplicada apenas aos FPs `conf>=0.50` ({fp_est['high_conf_fp_count']}/{fp_est['total_fp_count']}) | "
                f"{_fmt(fp_est['adjusted_applying_rate_only_to_high_conf_fps']['baseline']['precision_adjusted'])} | "
                f"{_fmt(fp_est['adjusted_applying_rate_only_to_high_conf_fps']['baseline']['recall_adjusted'])} | "
                f"{_fmt(fp_est['adjusted_applying_rate_only_to_high_conf_fps']['baseline']['f1_adjusted'])} | "
                f"{_fmt(fp_est['adjusted_applying_rate_only_to_high_conf_fps']['filtered']['precision_adjusted'])} | "
                f"{_fmt(fp_est['adjusted_applying_rate_only_to_high_conf_fps']['filtered']['recall_adjusted'])} | "
                f"{_fmt(fp_est['adjusted_applying_rate_only_to_high_conf_fps']['filtered']['f1_adjusted'])} | "
                f"{_fmt_signed(fp_est['adjusted_applying_rate_only_to_high_conf_fps']['delta_f1_adjusted'])} |"
            ),
            "",
            "## Procedência e auditoria do `soccernet_gsr_test_virgin` (descartado como prova)",
            "",
            f"- **Origem**: `{gsr_audit_note['source_path']}` (split `test` do SoccerNet Game State Reconstruction 2025, 49 clipes 1080p de transmissão).",
            f"- **Montagem na Rodada 1**: `{gsr_audit_note['construction']}`.",
            f"- **Prova de 0 sobreposição com `r1`**: `{gsr_audit_note['overlap_proof']}`.",
            f"- **Confiabilidade das anotações de bola (auditoria visual)**: {gsr_audit_note['reliability_verdict']}",
            "- **`box2boxcam` (câmera amadora/noturna, 198 quadros)**: não disponível localmente; permanece não medido.",
            "",
            "## Rodada 1 (preservada como referência histórica)",
            "",
            (
                f"Arquivo completo preservado em `results/rodada1_experiment_results.json`. Na Rodada 1, o protocolo usou "
                f"recorte justo `context_scale=1.0` (`48x96` letterbox), partição única 50/50 de `val.txt` (`val_calib` 1.624 imgs / 642 jogos, "
                f"F1 base `{_fmt(round1['baseline_calib']['f1'])}`; `val_eval` 1.623 imgs / 151 jogos contendo `v14`, F1 base `{_fmt(r1_base['f1'])}`), "
                f"mineração em 1.500 imagens de `train.txt` e 8 épocas × 180 episódios (1.440 episódios), com `{round1['protocol']['total_configs_tested']}` configurações testadas:"
            ),
            "",
            "| Braço (Rodada 1, `val_eval`) | Config selecionada | Precisão | Recall | F1 | mAP50 | ROC-AUC | FPs rem | TPs rem | ΔF1 (IC 95% imgs) |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            (
                f"| Detector sozinho | baseline | {_fmt(r1_base['precision'])} | {_fmt(r1_base['recall'])} | "
                f"{_fmt(r1_base['f1'])} | {_fmt(r1_base['map50'])} | {_fmt(round1['detector_conf_separability']['eval_roc_auc'])} (`conf`) | "
                f"0/{r1_base['fp']} | 0/{r1_base['tp']} | 0.0000 |"
            ),
        ]
    )
    for r1_id_key in ("selected_frozen_config_id", "selected_trained_config_id"):
        cid = round1.get(r1_id_key)
        if not cid:
            continue
        c = next(x for x in round1["configurations"] if x["config_id"] == cid)
        cm = c["eval_metrics"]
        cc = c["filter_cost"]
        cb = c["bootstrap"]["delta_f1"]
        lines.append(
            f"| {_arm_label(c['arm'])} | `{cid}` | {_fmt(cm['precision'])} | {_fmt(cm['recall'])} | "
            f"{_fmt(cm['f1'])} | {_fmt(cm['map50'])} | {_fmt(c['separability']['eval_roc_auc'])} | "
            f"{cc['fps_removed']}/{cc['base_fp']} | {cc['tps_removed']}/{cc['base_tp']} | "
            f"{_fmt_signed(cb['point_estimate'])} [{_fmt_signed(cb['ci_95_low'])}, {_fmt_signed(cb['ci_95_high'])}] |"
        )

    lines.extend(
        [
            "",
            "## Limitações",
            "",
            "- `val.txt` não é teste virgem do detector: foi usado para selecionar o checkpoint `t1b_v5_etapa1_rodada1.pt`. Os valores absolutos podem ser otimistas, embora a comparação entre os braços ocorra sob o mesmo viés.",
            "- O grupo `v14` concentra 1.111 das 3.247 imagens de `val.txt` (34,2%) sob um único `game_id`, aumentando a variância do bootstrap por jogo e criando uma dobra maior na CV de 5 dobras.",
            "- O filtro só remove detecções (nunca gera novas caixas nem aumenta recall); quando o detector já tem precisão próxima de 0,75–0,80, qualquer remoção imprecisa custa caro em F1.",
            "- Parte dos FPs registrados em `val.txt` corresponde a bolas reais não anotadas (auditado em `results/amostra_fp.png`), o que penaliza tanto o detector quanto qualquer filtro que acerte ao manter essas bolas.",
            "- A mineração completa em `train.txt` (32.852 imagens) usou `imgsz=1280` (resolução de treino do `t1b_v5`) por viabilidade computacional no Mac MPS, enquanto a avaliação oficial em `val.txt` manteve `imgsz=1920` (padrão do site).",
            "- Não havia conjunto virgem confiável de câmera amadora/noturna (`box2boxcam`, 198 quadros) disponível localmente; `soccernet_gsr_test_virgin` foi auditado visualmente e descartado por caixas GT de bola deslocadas.",
            "",
        ]
    )
    return "\n".join(lines)
