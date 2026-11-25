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


def _direction_short_pt(direction: str) -> str:
    if direction == "improvement":
        return "melhora"
    if direction == "degradation":
        return "piora"
    return "indistinguível"


def _direction_pt(direction: str) -> str:
    if direction == "improvement":
        return "melhora distinguível de zero"
    if direction == "degradation":
        return "piora distinguível de zero"
    return "indistinguível de zero"


def render_report(results: dict[str, Any]) -> str:
    protocol = results["protocol"]
    splits = results["splits"]
    base = results["baseline_eval"]
    configs = results["configurations"]
    selected_id = results["selected_config_id"]
    selected = next(c for c in configs if c["config_id"] == selected_id)
    sel_eval = selected["eval_metrics"]
    sel_cost = selected["filter_cost"]
    sel_boot = selected["bootstrap"]["delta_f1"]
    sel_sep = selected["separability"]
    headroom = base["headroom"]

    verdict_delta = sel_boot["point_estimate"]
    verdict_text = _direction_pt(sel_boot["direction"])
    if selected["calibration"]["is_noop"]:
        verdict_line = (
            f"1. Veredito: o filtro congelado ({selected_id}) não aplicou corte (limiar -inf na calibração); "
            f"nenhuma detecção foi removida no conjunto final."
        )
    else:
        verdict_line = (
            f"1. Veredito: no conjunto final ({splits['eval_split_name']}, n={splits['eval_images']} imagens), "
            f"a configuração escolhida na calibração ({selected_id}) variou o F1 em {_fmt_signed(verdict_delta)} "
            f"({verdict_text})."
        )

    extra_datasets = results.get("extra_datasets") or []
    box2box = next((d for d in extra_datasets if "box2box" in d["name"].lower()), None)
    if box2box is not None:
        b2b_boot = box2box["bootstrap"]["delta_f1"]
        extra_line = (
            f"7. Câmera amadora ({box2box['name']}, n={box2box['baseline']['n_images']} imagens): "
            f"ΔF1={_fmt_signed(b2b_boot['point_estimate'])} "
            f"(IC 95% [{_fmt_signed(b2b_boot['ci_95_low'])}, {_fmt_signed(b2b_boot['ci_95_high'])}], "
            f"ROC-AUC={_fmt(box2box['separability']['roc_auc'])})."
        )
    else:
        extra_line = "7. Câmera amadora (box2boxcam): não medido (conjunto não fornecido ou inacessível nesta execução)."

    summary_lines = [
        verdict_line,
        (
            f"2. Detector sozinho ({protocol['detector_weights']}, imgsz={protocol['imgsz']}, conf={protocol['conf']}): "
            f"Precisão={_fmt(base['precision'])}, Recall={_fmt(base['recall'])}, F1={_fmt(base['f1'])}, "
            f"mAP50={_fmt(base['map50'])}, mAP50-95={_fmt(base['map50_95'])}."
        ),
        (
            f"3. Detector + FewShot-R18 ({selected_id}, limiar congelado={_fmt_threshold(selected['calibration']['threshold'])}): "
            f"Precisão={_fmt(sel_eval['precision'])}, Recall={_fmt(sel_eval['recall'])}, F1={_fmt(sel_eval['f1'])}, "
            f"mAP50={_fmt(sel_eval['map50'])}, mAP50-95={_fmt(sel_eval['map50_95'])}."
        ),
        (
            f"4. Custo real do filtro: removeu {sel_cost['fps_removed']}/{sel_cost['base_fp']} falsos positivos "
            f"({_fmt(sel_cost['fp_removal_rate'] * 100, 1)}%) e removeu {sel_cost['tps_removed']}/{sel_cost['base_tp']} "
            f"bolas verdadeiras por engano (ΔRecall={_fmt_signed(sel_eval['recall'] - base['recall'])})."
        ),
        (
            f"5. Intervalo bootstrap em ΔF1 ({protocol['bootstrap_resamples']} reamostragens pareadas sobre imagens): "
            f"IC 95% [{_fmt_signed(sel_boot['ci_95_low'])}, {_fmt_signed(sel_boot['ci_95_high'])}] — {verdict_text}."
        ),
        (
            f"6. Separabilidade do FewShot-R18 (bola vs FP, sem limiar): "
            f"ROC-AUC={_fmt(sel_sep['eval_roc_auc'])} no eval ({_fmt(sel_sep['calib_roc_auc'])} na calibração); "
            f"PR-AUC={_fmt(sel_sep['eval_pr_auc'])} no eval."
        ),
        extra_line,
        (
            f"8. Teto de ganho possível do detector sozinho: {base['fp']} FPs em {base['n_images']} imagens "
            f"({base['fp_per_image']:.3f} FP/imagem), teto de precisão={_fmt_signed(headroom['precision_headroom'])} "
            f"e teto de F1 sem novo recall={_fmt_signed(headroom['f1_headroom'])}; "
            f"total de configurações testadas={protocol['total_configs_tested']}."
        ),
    ]

    lines: list[str] = [
        "# Relatório: Detector de bola + filtro de falsos positivos FewShot-R18",
        "",
        "## Resumo (8 linhas)",
        "",
        *summary_lines,
        "",
        "## Comparação principal no conjunto de avaliação",
        "",
        (
            f"Detector: `{protocol['detector_weights']}` | Filtro: `{protocol['fewshot_weights']}` | "
            f"imgsz={protocol['imgsz']} | conf={protocol['conf']} | IoU={protocol['eval_iou']} | "
            f"Split de avaliação: `{splits['eval_split_name']}` ({splits['eval_images']} imagens, {splits['eval_gt_balls']} bolas GT)."
        ),
        "",
        "| Configuração | Estágio | K | Limiar congelado | Precisão | Recall | F1 | mAP50 | mAP50-95 | FPs removidos | TPs removidos | ΔF1 (IC 95%) | Distinguível de zero? |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        (
            f"| Detector sozinho (baseline) | baseline | - | - | {_fmt(base['precision'])} | "
            f"{_fmt(base['recall'])} | {_fmt(base['f1'])} | {_fmt(base['map50'])} | "
            f"{_fmt(base['map50_95'])} | 0 / {base['fp']} | 0 / {base['tp']} | 0.0000 | - |"
        ),
    ]

    for cfg in configs:
        em = cfg["eval_metrics"]
        cost = cfg["filter_cost"]
        bf1 = cfg["bootstrap"]["delta_f1"]
        marker = " *" if cfg["config_id"] == selected_id else ""
        dist_label = "Sim" if bf1["distinguishable_from_zero"] else "Não"
        lines.append(
            f"| `{cfg['config_id']}`{marker} | {cfg['stage']} | {cfg['k_pos_used']} | "
            f"{_fmt_threshold(cfg['calibration']['threshold'])} | {_fmt(em['precision'])} | "
            f"{_fmt(em['recall'])} | {_fmt(em['f1'])} | {_fmt(em['map50'])} | "
            f"{_fmt(em['map50_95'])} | {cost['fps_removed']} / {cost['base_fp']} | "
            f"{cost['tps_removed']} / {cost['base_tp']} | "
            f"{_fmt_signed(bf1['point_estimate'])} [{_fmt_signed(bf1['ci_95_low'])}, {_fmt_signed(bf1['ci_95_high'])}] | "
            f"{dist_label} ({_direction_short_pt(bf1['direction'])}) |"
        )

    lines.extend(
        [
            "",
            "`*` Configuração selecionada exclusivamente pela métrica F1 na calibração antes de abrir o conjunto final.",
            "",
            "## O FewShot-R18 separa bola de falso positivo?",
            "",
            (
                "Antes de escolher qualquer limiar, medimos se o escore de similaridade do FewShot-R18 "
                "(treinado originalmente para 4 classes de jogadores/goleiros, não para bola) ordena detecções "
                "verdadeiras (TP) acima de falsos positivos (FP)."
            ),
            "",
            "| Configuração | Camada | Modo | K pos / neg | ROC-AUC (calib) | PR-AUC (calib) | ROC-AUC (eval) | PR-AUC (eval) | TP / FP (eval) |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )

    for cfg in configs:
        sep = cfg["separability"]
        lines.append(
            f"| `{cfg['config_id']}` | `{cfg['layer']}` | `{cfg['mode']}` | "
            f"{cfg['k_pos_used']} / {cfg['k_neg_used']} | {_fmt(sep['calib_roc_auc'])} | "
            f"{_fmt(sep['calib_pr_auc'])} | {_fmt(sep['eval_roc_auc'])} | "
            f"{_fmt(sep['eval_pr_auc'])} | {sep['eval_n_tp']} / {sep['eval_n_fp']} |"
        )

    lines.extend(
        [
            "",
            "## Teto de ganho possível (headroom)",
            "",
            (
                "Como o filtro só remove detecções (nunca cria novas caixas nem recupera bolas perdidas), "
                "mesmo um filtro oráculo perfeito (100% dos FPs removidos e 0 TPs perdidos) tem ganho limitado "
                "pelo volume de FPs do detector base:"
            ),
            "",
            f"- Detecções base no eval: {base['total_pred']} ({base['tp']} TP, {base['fp']} FP, {base['fn']} FN; {base['fp_per_image']:.3f} FP/imagem).",
            f"- Precisão base: {_fmt(base['precision'])} → teto teórico: {_fmt(headroom['max_precision_perfect_filter'])} (ganho máximo de precisão: {_fmt_signed(headroom['precision_headroom'])}).",
            f"- F1 base: {_fmt(base['f1'])} → teto teórico sem ganho de recall: {_fmt(headroom['max_f1_perfect_filter'])} (ganho máximo de F1: {_fmt_signed(headroom['f1_headroom'])}).",
            "",
            "## Protocolo, higiene de splits e escolhas",
            "",
            f"- Fonte dos splits: `{protocol['data_yaml']}` (lista de validação: `{splits.get('val_list_source', 'data.yaml:val')}`; disjunção treino/val verificada: `{splits['train_val_disjoint_verified']}`).",
            f"- Referências do filtro: sorteadas automaticamente apenas do treino (`train_ref`: {splits['train_ref_images']} imagens, {splits['train_ref_gt_balls']} bolas GT) com seed fixa `{protocol['seed']}`.",
            f"- Calibração do limiar: `{protocol['calib_source']}` (`{splits['calib_split_name']}`, {splits['calib_images']} imagens, fração `{protocol['calib_fraction']}`). Limiar congelado antes de tocar `{splits['eval_split_name']}` ({splits['eval_images']} imagens).",
            f"- Pré-processamento do crop: canvas letterbox `{protocol['canvas_wh']}`, `context_scale={protocol['context_scale']}`, `square_crop={protocol['square_crop']}`.",
            f"- Configurações testadas: {protocol['total_configs_tested']} (política de variações `{protocol['variations_policy']}`, gatilho AUC `< {protocol['auc_variation_trigger']}`, variações executadas: `{protocol['variations_executed']}`).",
            f"- Escolha registrada: `{selected_id}` — {results['selection_rationale']}.",
        ]
    )

    val_sampling = splits.get("val_sampling") or {}
    if val_sampling.get("subsampled"):
        lines.append(
            f"- Amostragem estratificada de `val`: {val_sampling['sampled_images']}/{val_sampling['original_images']} "
            f"imagens (seed `{val_sampling['seed']}`, {val_sampling.get('n_strata', '?')} estratos, "
            f"{val_sampling.get('sampled_pos_images', '?')}/{val_sampling.get('original_pos_images', '?')} imagens com bola)."
        )

    if protocol["skipped_variants"]:
        lines.append("")
        lines.append("Variações não executadas:")
        for skipped in protocol["skipped_variants"]:
            lines.append(f"- `{skipped['variant']}`: {skipped['reason']}")

    lines.extend(["", "## Câmera amadora (conjuntos extras)", ""])
    if not extra_datasets:
        lines.append(
            "Não medido: nenhum conjunto extra (como `box2boxcam`) foi fornecido ou estava acessível nesta execução."
        )
    else:
        for ds in extra_datasets:
            db = ds["baseline"]
            df = ds["filtered"]
            dc = ds["filter_cost"]
            dboot = ds["bootstrap"]["delta_f1"]
            dsep = ds["separability"]
            lines.extend(
                [
                    f"### {ds['name']}",
                    "",
                    f"- Configuração congelada aplicada (sem reajuste): `{ds['applied_config_id']}`, limiar `{_fmt_threshold(ds['frozen_threshold'])}`.",
                    f"- Baseline: Precisão={_fmt(db['precision'])}, Recall={_fmt(db['recall'])}, F1={_fmt(db['f1'])}, mAP50={_fmt(db['map50'])}, mAP50-95={_fmt(db['map50_95'])} ({db['tp']} TP, {db['fp']} FP em {db['n_images']} imagens).",
                    f"- Filtrado: Precisão={_fmt(df['precision'])}, Recall={_fmt(df['recall'])}, F1={_fmt(df['f1'])}, mAP50={_fmt(df['map50'])}, mAP50-95={_fmt(df['map50_95'])}.",
                    f"- Custo: {dc['fps_removed']}/{dc['base_fp']} FPs removidos, {dc['tps_removed']}/{dc['base_tp']} TPs removidos; ROC-AUC={_fmt(dsep['roc_auc'])}, PR-AUC={_fmt(dsep['pr_auc'])}.",
                    f"- ΔF1={_fmt_signed(dboot['point_estimate'])}, IC 95% [{_fmt_signed(dboot['ci_95_low'])}, {_fmt_signed(dboot['ci_95_high'])}] ({_direction_pt(dboot['direction'])}).",
                    "",
                ]
            )

    lines.extend(["", "## Limitações", ""])
    limitations = [
        "O FewShot-R18 foi treinado para classificar uniformes de jogadores e goleiros (4 classes), não para distinguir bola de fundo/objetos; qualquer separação depende de transferência fora de domínio.",
        "O filtro atua apenas em pós-processamento (remoção de FPs) e nunca aumenta recall nem recupera bolas não propostas pelo detector.",
        "Bolas em quadros 1080p/4K costumam ter poucos pixels; o redimensionamento de crops muito pequenos pode perder textura e forma.",
        "O protótipo few-shot depende da semente fixa de amostragem das referências de treino; variar a semente pode alterar o resultado especialmente em k=1.",
        "Não há garantia de generalização entre câmeras profissionais/estádio e câmera amadora/noturna (como box2boxcam) sem medição direta nesse domínio.",
    ]
    for warning in results.get("warnings") or []:
        limitations.append(f"Aviso desta execução: {warning}.")
    for item in limitations:
        lines.append(f"- {item}")

    lines.append("")
    return "\n".join(lines)
