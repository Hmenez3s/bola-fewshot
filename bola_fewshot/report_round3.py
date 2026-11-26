from __future__ import annotations

from typing import Any

from bola_fewshot.report_round2 import render_combined_report


def _fmt(value: float | None, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def _fmt_signed(value: float | None, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    return f"{value:+.{digits}f}"


def _dir_short(direction: str) -> str:
    if direction == "improvement":
        return "melhora"
    if direction == "degradation":
        return "piora"
    return "indistinguível"


def _best_method_across_ops(round3: dict[str, Any], scope: str = "all_val") -> tuple[str, str, dict[str, Any]]:
    best_op = ""
    best_name = ""
    best_entry: dict[str, Any] = {}
    best_f1 = -1.0
    for op_key, op_val in round3["operating_points"].items():
        for mname, mval in op_val["methods"].items():
            sub = mval[scope]
            f1 = sub["filtered"]["f1"]
            if f1 > best_f1:
                best_f1 = f1
                best_op = op_key
                best_name = mname
                best_entry = sub
    return best_op, best_name, best_entry


def _best_cheap_across_ops(round3: dict[str, Any], scope: str = "all_val") -> tuple[str, str, dict[str, Any]]:
    best_op = ""
    best_name = ""
    best_entry: dict[str, Any] = {}
    best_f1 = -1.0
    for op_key, op_val in round3["operating_points"].items():
        for cname, cval in op_val["cheap_baselines"].items():
            sub = cval[scope]
            f1 = sub["filtered"]["f1"]
            if f1 > best_f1:
                best_f1 = f1
                best_op = op_key
                best_name = cname
                best_entry = sub
    return best_op, best_name, best_entry


def render_three_rounds_report(
    round1: dict[str, Any],
    round2: dict[str, Any],
    round3: dict[str, Any],
    fp_audit_report: dict[str, Any],
    gsr_audit_note: dict[str, Any],
) -> str:
    p3 = round3["protocol"]
    cov = round3["auto_reference_audit"]
    ops = round3["operating_points"]
    op35 = ops["conf_0.35"]
    base35_all = op35["baseline_all_val"]

    best_op_all, best_m_all, best_sub_all = _best_method_across_ops(round3, "all_val")
    best_cop_all, best_c_all, best_csub_all = _best_cheap_across_ops(round3, "all_val")
    best_op_el, best_m_el, best_sub_el = _best_method_across_ops(round3, "eligible_queries")

    bm_f1 = best_sub_all["filtered"]["f1"]
    bc_f1 = best_csub_all["filtered"]["f1"]
    bm_df1_vs_35 = bm_f1 - base35_all["f1"]
    bc_df1_vs_35 = bc_f1 - base35_all["f1"]

    r1_tr_id = round1["selected_trained_config_id"]
    r1_tr = next(c for c in round1["configurations"] if c["config_id"] == r1_tr_id)
    r1_df1 = r1_tr["bootstrap"]["delta_f1"]["point_estimate"]
    r2_l34 = round2["nested_arms"]["trained_ball_layer34"]
    r2_df1 = r2_l34["bootstrap_group"]["delta_f1"]["point_estimate"]

    m35_log = op35["methods"]["trained_per_game_logistic_fusion"]["all_val"]
    m35_log_t1 = op35["methods"]["trained_per_game_logistic_plus_top1"]["all_val"]
    m35_top1 = op35["cheap_baselines"]["top1_per_frame"]["all_val"]
    m35_cut_el = op35["methods"]["trained_per_game_cut_keep_fallback"]["eligible_queries"]
    m35_log_el = op35["methods"]["trained_per_game_logistic_fusion"]["eligible_queries"]
    m35_frz_el = op35["methods"]["frozen_per_game_cut"]["eligible_queries"]

    best_bg = best_sub_all["bootstrap_group"]["delta_f1"]
    best_bi = best_sub_all["bootstrap_image"]["delta_f1"]
    m35_lt_bg = m35_log_t1["bootstrap_group"]["delta_f1"]

    op15_base = ops.get("conf_0.15", {}).get("baseline_all_val")
    op15_log = ops.get("conf_0.15", {}).get("methods", {}).get("trained_per_game_logistic_plus_top1", {}).get("all_val")
    op10_base = ops.get("conf_0.10", {}).get("baseline_all_val")
    op10_log = ops.get("conf_0.10", {}).get("methods", {}).get("trained_per_game_logistic_plus_top1", {}).get("all_val")

    modes35 = op35.get("mode_breakdown_eligible", {})
    mode_parts = []
    for tag, label in (
        ("positive_only", "pos_only"),
        ("pos_minus_neg_same_game", "pos−neg_jogo"),
        ("pos_minus_neg_global", "pos−neg_global"),
    ):
        if tag in modes35:
            mv = modes35[tag]
            df = mv["eligible_queries"]["bootstrap_group"]["delta_f1"]["point_estimate"]
            mode_parts.append(f"{label}: ΔF1={_fmt_signed(df)} (AUC={_fmt(mv['mean_fold_roc_auc_elig'])})")

    low_conf_text = ""
    if op15_base and op15_log and op10_base and op10_log:
        low_conf_text = (
            f"Baixar conf para 0.15 (R base={_fmt(op15_base['recall'])}, F1={_fmt(op15_base['f1'])}) + `{best_m_all}` "
            f"chegou a F1={_fmt(op15_log['filtered']['f1'])} (R={_fmt(op15_log['filtered']['recall'])}, P={_fmt(op15_log['filtered']['precision'])}); "
            f"em conf=0.10 (R base={_fmt(op10_base['recall'])}, F1={_fmt(op10_base['f1'])}) chegou a F1={_fmt(op10_log['filtered']['f1'])}."
        )

    fp_est = fp_audit_report["noise_adjustment"]
    summary_lines = [
        (
            f"1. Veredito (Rodada 1 → 2 → 3): R1 (ctx 1×, 50/50) ΔF1={_fmt_signed(r1_df1)}; R2 (ctx 2×, global, CV 5 dobras) ΔF1={_fmt_signed(r2_df1)}; "
            f"R3 (por jogo + fusão + recall alto): melhor método FewShot `{best_op_all}/{best_m_all}` atingiu F1={_fmt(bm_f1)} "
            f"({_fmt_signed(bm_df1_vs_35)} sobre o detector padrão conf=0.35 F1={_fmt(base35_all['f1'])}), contra melhor baseline barato `{best_cop_all}/{best_c_all}` F1={_fmt(bc_f1)} ({_fmt_signed(bc_df1_vs_35)})."
        ),
        (
            f"2. Auto-referências por jogo (sem GT, conf>={cov['ref_min_conf']}): cobertura={cov['eligible_games']}/{cov['total_games']} jogos "
            f"({cov['eligible_games_pct']*100:.1f}% dos jogos, {cov['eligible_total_images']}/{cov['total_images']} imagens = {cov['eligible_total_images']/cov['total_images']*100:.1f}%; "
            f"{cov['eligible_no_v14_games']} jogos reais sem `v14`); {cov['eligible_ref_images_excluded']} quadros de referência excluídos da avaliação; "
            f"precisão real das auto-referências={cov['auto_ref_precision_iou50']*100:.1f}% (IoU>=0.50) e {cov['auto_ref_precision_iou30']*100:.1f}% (IoU>=0.30)."
        ),
        (
            f"3. Em conf=0.35 (padrão do site, todo val.txt): Base F1={_fmt(base35_all['f1'])} | top1_per_frame F1={_fmt(m35_top1['filtered']['f1'])} | "
            f"fusão logística por jogo F1={_fmt(m35_log['filtered']['f1'])} | fusão logística + top1 F1={_fmt(m35_log_t1['filtered']['f1'])} "
            f"(ΔF1={_fmt_signed(m35_lt_bg['point_estimate'])}, IC jogos [{_fmt_signed(m35_lt_bg['ci_95_low'])}, {_fmt_signed(m35_lt_bg['ci_95_high'])}])."
        ),
        f"4. Ponto de recall alto (conf=0.15 e 0.10): {low_conf_text}",
        (
            f"5. Melhor resultado no subconjunto elegível (`{best_op_el}/{best_m_el}`): F1={_fmt(best_sub_el['filtered']['f1'])} vs base `{best_op_el}`={_fmt(best_sub_el['baseline']['f1'])} "
            f"(ΔF1={_fmt_signed(best_sub_el['bootstrap_group']['delta_f1']['point_estimate'])}, IC jogos [{_fmt_signed(best_sub_el['bootstrap_group']['delta_f1']['ci_95_low'])}, {_fmt_signed(best_sub_el['bootstrap_group']['delta_f1']['ci_95_high'])}], "
            f"FPs rem={best_sub_el['fps_removed']}, TPs rem={best_sub_el['tps_removed']}); em `all_val` `{best_op_all}/{best_m_all}`: IC jogos [{_fmt_signed(best_bg['ci_95_low'])}, {_fmt_signed(best_bg['ci_95_high'])}], IC imgs [{_fmt_signed(best_bi['ci_95_low'])}, {_fmt_signed(best_bi['ci_95_high'])}]."
        ),
        (
            f"6. Modos por jogo (conf=0.35, elegível, corte seco): " + "; ".join(mode_parts)
            + f"; controle congelado por jogo ΔF1={_fmt_signed(m35_frz_el['bootstrap_group']['delta_f1']['point_estimate'])}, "
            + f"fusão logística elegível ΔF1={_fmt_signed(m35_log_el['bootstrap_group']['delta_f1']['point_estimate'])}."
        ),
        (
            f"7. Custo real em `{best_op_all}/{best_m_all}` (todo val.txt): P={_fmt(best_sub_all['filtered']['precision'])}, R={_fmt(best_sub_all['filtered']['recall'])}, "
            f"mAP50={_fmt(best_sub_all['filtered']['map50'])} (vs base 0.35 mAP50={_fmt(base35_all['map50'])}); "
            f"removeu {best_sub_all['fps_removed']}/{best_sub_all['baseline']['fp']} FPs e perdeu {best_sub_all['tps_removed']}/{best_sub_all['baseline']['tp']} TPs relativamente à base `{best_op_all}`."
        ),
        (
            f"8. Viés e auditorias: `val.txt` não é teste virgem do detector (`train∩val=0` jogos); em `results/amostra_fp.png` (40 FPs conf>=0.50), "
            f"55,0% são quase-acertos/duplicados da bola anotada, {fp_est['unannotated_ball_rate_strict']*100:.1f}% bolas extras não anotadas e 17,5% FPs não-bola; "
            f"`soccernet_gsr` descartado (GT deslocado), `box2boxcam` indisponível; configs testadas: R1={round1['protocol']['total_configs_tested']}, R2={round2['protocol']['total_fixed_configs_tested']}+3, R3={p3['total_configs_tested']}."
        ),
    ]

    lines: list[str] = [
        "# Relatório: Detector de bola + filtro FewShot-R18 (Rodadas 1, 2 e 3)",
        "",
        "## Resumo (8 linhas — Rodada 1 vs Rodada 2 vs Rodada 3)",
        "",
        *summary_lines,
        "",
        "## Rodada 3 — Few-shot por jogo (6 auto-referências sem GT), fusão com `conf`, recall alto e NMS/top-1",
        "",
        (
            f"Detector: `{p3['detector_weights']}` (`imgsz={p3['imgsz']}`, `IoU={p3['eval_iou']}`) | "
            f"Pontos operacionais testados: `conf ∈ {p3['operating_confs']}` | Contexto quadrado `{p3['canvas_wh'][0]}x{p3['canvas_wh'][1]}` com `context_scale ∈ {p3['context_scales']}` | "
            f"Treino episódico POR JOGO (`layer3/4`, `{p3['train_epochs']}` épocas × `{p3['train_steps_per_epoch']}` episódios = `{p3['train_total_episodes']}`, cap `{p3['train_cap_per_game']}` imgs/jogo cobrindo os 488 jogos de `train.txt`) | "
            f"CV aninhada de `{p3['n_folds']}` dobras por jogo sobre `val.txt` (tudo congelado na calibração antes de avaliar cada dobra; aviso: `val.txt` não é teste virgem do detector)."
        ),
        "",
        "### Cobertura e precisão real das 6 auto-referências por jogo (sem uso de GT)",
        "",
        f"- Regra de seleção: até `{cov['k_refs_required']}` detecções do próprio jogo com `conf >= {cov['ref_min_conf']}`, 1 por quadro e espalhadas em `{cov['k_refs_required']}` faixas uniformes dos quadros do jogo; os quadros escolhidos como referência são **excluídos** da avaliação de consulta.",
        f"- Cobertura de jogos: **`{cov['eligible_games']} / {cov['total_games']}` jogos** ({cov['eligible_games_pct']*100:.1f}% dos jogos), correspondendo a **`{cov['eligible_total_images']} / {cov['total_images']}` imagens** ({cov['eligible_total_images']/cov['total_images']*100:.1f}% do `val.txt`). Os demais `{cov['total_games'] - cov['eligible_games']}` jogos têm poucos quadros (735 jogos têm apenas 1 imagem).",
        f"- Quadros de referência excluídos: `{cov['eligible_ref_images_excluded']}` (`6 × {cov['eligible_games']}`). Quadros de consulta avaliados no subconjunto elegível (A): `{cov['eligible_query_images']}` imagens (`{cov['eligible_no_v14_query_images']}` imagens em `{cov['eligible_no_v14_games']}` jogos reais excluindo `v14`).",
        f"- **Precisão real das `{cov['auto_ref_total']}` auto-referências contra o GT**: **`{cov['auto_ref_tp_iou50']}/{cov['auto_ref_total']}` ({cov['auto_ref_precision_iou50']*100:.2f}%) em IoU>=0.50** e **`{cov['auto_ref_tp_iou30']}/{cov['auto_ref_total']}` ({cov['auto_ref_precision_iou30']*100:.2f}%) em IoU>=0.30**.",
        f"- Negativos de baixa confiança (`0.05..{p3['same_game_neg_max_conf']}`) coletados nos mesmos quadros de referência (sem tocar quadros de consulta): `{cov['same_game_neg_refs_found']}` recortes (taxa limpa `IoU<0.10` contra GT: `{cov['same_game_neg_clean_rate']*100:.1f}%`).",
        "",
    ]

    for op_conf in p3["operating_confs"]:
        op_key = f"conf_{op_conf:.2f}"
        op_val = ops[op_key]
        lines.extend(
            [
                f"### Ponto operacional `conf = {op_conf:.2f}`",
                "",
                f"Separabilidade da `conf` do YOLO neste corte: ROC-AUC `all_val`=`{_fmt(op_val['detector_conf_roc_auc_all'])}`, elegível=`{_fmt(op_val['detector_conf_roc_auc_elig'])}`.",
                "",
                "#### (B) Todo o `val.txt` operacional (3.247 imagens, 793 jogos)",
                "",
                "| Método | Tipo | Precisão | Recall | F1 | mAP50 | FPs rem | TPs rem | ΔF1 vs base deste conf (IC 95% jogos) | ΔF1 vs base 0.35 |",
                "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        b_all = op_val["baseline_all_val"]
        lines.append(
            f"| Detector sozinho (`conf={op_conf:.2f}`, `iou=0.70`) | baseline | {_fmt(b_all['precision'])} | {_fmt(b_all['recall'])} | "
            f"{_fmt(b_all['f1'])} | {_fmt(b_all['map50'])} | 0/{b_all['fp']} | 0/{b_all['tp']} | 0.0000 | {_fmt_signed(b_all['f1'] - base35_all['f1'])} |"
        )
        for cname, cpack in op_val["cheap_baselines"].items():
            sub = cpack["all_val"]
            fm = sub["filtered"]
            bg = sub["bootstrap_group"]["delta_f1"]
            lines.append(
                f"| `{cname}` | barato (sem rede) | {_fmt(fm['precision'])} | {_fmt(fm['recall'])} | {_fmt(fm['f1'])} | {_fmt(fm['map50'])} | "
                f"{sub['fps_removed']}/{sub['baseline']['fp']} | {sub['tps_removed']}/{sub['baseline']['tp']} | "
                f"{_fmt_signed(bg['point_estimate'])} [{_fmt_signed(bg['ci_95_low'])}, {_fmt_signed(bg['ci_95_high'])}] ({_dir_short(bg['direction'])}) | "
                f"{_fmt_signed(fm['f1'] - base35_all['f1'])} |"
            )
        for mname, mpack in op_val["methods"].items():
            sub = mpack["all_val"]
            fm = sub["filtered"]
            bg = sub["bootstrap_group"]["delta_f1"]
            tag = f"{mpack['strategy']} / {mpack['fallback_mode']}" + (" + top1" if mpack["top1_after"] else "")
            lines.append(
                f"| `{mname}` | {tag} | {_fmt(fm['precision'])} | {_fmt(fm['recall'])} | {_fmt(fm['f1'])} | {_fmt(fm['map50'])} | "
                f"{sub['fps_removed']}/{sub['baseline']['fp']} | {sub['tps_removed']}/{sub['baseline']['tp']} | "
                f"{_fmt_signed(bg['point_estimate'])} [{_fmt_signed(bg['ci_95_low'])}, {_fmt_signed(bg['ci_95_high'])}] ({_dir_short(bg['direction'])}) | "
                f"{_fmt_signed(fm['f1'] - base35_all['f1'])} |"
            )

        lines.extend(
            [
                "",
                f"#### (A) Subconjunto elegível de jogos com 6 auto-referências ({cov['eligible_games']} jogos, {cov['eligible_query_images']} quadros de consulta; quadros de referência excluídos)",
                "",
                "| Método | AUC média/dobra | Precisão | Recall | F1 | mAP50 | FPs rem | TPs rem | ΔF1 elegível (IC 95% jogos) | ΔF1 sem `v14` (32 jogos) |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        b_el = op_val["baseline_eligible_queries"]
        b_nv = op_val.get("baseline_eligible_no_v14")
        lines.append(
            f"| Detector sozinho (`conf={op_conf:.2f}`) | {_fmt(op_val['detector_conf_roc_auc_elig'])} (`conf`) | "
            f"{_fmt(b_el['precision'])} | {_fmt(b_el['recall'])} | {_fmt(b_el['f1'])} | {_fmt(b_el['map50'])} | "
            f"0/{b_el['fp']} | 0/{b_el['tp']} | 0.0000 | 0.0000 (base={_fmt(b_nv['f1'] if b_nv else None)}) |"
        )
        for cname, cpack in op_val["cheap_baselines"].items():
            sub = cpack["eligible_queries"]
            fm = sub["filtered"]
            bg = sub["bootstrap_group"]["delta_f1"]
            lines.append(
                f"| `{cname}` | - | {_fmt(fm['precision'])} | {_fmt(fm['recall'])} | {_fmt(fm['f1'])} | {_fmt(fm['map50'])} | "
                f"{sub['fps_removed']}/{sub['baseline']['fp']} | {sub['tps_removed']}/{sub['baseline']['tp']} | "
                f"{_fmt_signed(bg['point_estimate'])} [{_fmt_signed(bg['ci_95_low'])}, {_fmt_signed(bg['ci_95_high'])}] | - |"
            )
        for mname, mpack in op_val["methods"].items():
            sub = mpack["eligible_queries"]
            fm = sub["filtered"]
            bg = sub["bootstrap_group"]["delta_f1"]
            nv_sub = mpack.get("eligible_no_v14_queries")
            nv_str = (
                f"{_fmt_signed(nv_sub['bootstrap_group']['delta_f1']['point_estimate'])} "
                f"[{_fmt_signed(nv_sub['bootstrap_group']['delta_f1']['ci_95_low'])}, {_fmt_signed(nv_sub['bootstrap_group']['delta_f1']['ci_95_high'])}]"
                if nv_sub
                else "n/a"
            )
            lines.append(
                f"| `{mname}` | {_fmt(mpack['mean_fold_roc_auc_elig'])} | {_fmt(fm['precision'])} | {_fmt(fm['recall'])} | "
                f"{_fmt(fm['f1'])} | {_fmt(fm['map50'])} | {sub['fps_removed']}/{sub['baseline']['fp']} | "
                f"{sub['tps_removed']}/{sub['baseline']['tp']} | {_fmt_signed(bg['point_estimate'])} "
                f"[{_fmt_signed(bg['ci_95_low'])}, {_fmt_signed(bg['ci_95_high'])}] | {nv_str} |"
            )

        if op_val.get("mode_breakdown_eligible"):
            lines.extend(["", "Comparação de modos de protótipo por jogo (corte seco no subconjunto elegível):", ""])
            for mtag, mres in op_val["mode_breakdown_eligible"].items():
                eq = mres["eligible_queries"]
                bg = eq["bootstrap_group"]["delta_f1"]
                lines.append(
                    f"- `{mtag}`: AUC média/dobra=`{_fmt(mres['mean_fold_roc_auc_elig'])}`, "
                    f"F1=`{_fmt(eq['filtered']['f1'])}` (ΔF1=`{_fmt_signed(bg['point_estimate'])}`, "
                    f"IC 95% jogos `[{_fmt_signed(bg['ci_95_low'])}, {_fmt_signed(bg['ci_95_high'])}]`, "
                    f"FPs rem={eq['fps_removed']}/{eq['baseline']['fp']}, TPs rem={eq['tps_removed']}/{eq['baseline']['tp']})."
                )
        lines.append("")

    lines.extend(
        [
            "### Treino episódico por jogo no `train.txt` (Rodada 3)",
            "",
            (
                f"Mineração balanceada por jogo (`cap={p3['train_cap_per_game']}` imgs/jogo cobrindo todos os 488 jogos de `train.txt`: "
                f"`{round3['mining_stats']['train_images_scanned']}` imagens varridas a `imgsz={p3['train_mining_imgsz']}`, `conf>=0.05`). "
                f"Episódios montados com suporte (6 bolas + 6 negativos) e consultas (8 bolas + 8 negativos) do **mesmo jogo**, "
                f"com `layer3/4` treináveis, `BN` em `eval`, `AdamW(2e-4, wd=1e-4)`, `CosineAnnealingLR`, `cosine * 20`, `{p3['train_total_episodes']}` episódios."
            ),
            "",
            "| Contexto | Jogos elegíveis no treino | Recortes pos / neg mesmo jogo | % negativos do mesmo jogo nos episódios | Loss época 1 | Loss época final | Tempo (s) |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for tr in round3["training_runs"]:
        ps = tr["pool_stats"]
        hist = tr["history"]
        lines.append(
            f"| {tr['context_scale']:.0f}× | {ps['eligible_train_games']} | "
            f"{ps['total_pos_crops']} / {ps['total_same_game_neg_crops']} | "
            f"{tr['same_game_neg_fraction']*100:.1f}% | {hist[0]['loss']:.4f} | {hist[-1]['loss']:.4f} | {tr['total_train_s']:.1f} |"
        )

    r12_full = render_combined_report(round1, round2, fp_audit_report, gsr_audit_note)
    r2_body = r12_full.split("## Rodada 2 — ", 1)[1]
    lines.extend(["", "---", "", "## Rodada 2 — " + r2_body])
    return "\n".join(lines)
