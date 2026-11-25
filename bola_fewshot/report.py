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


def _arm_label(arm: str) -> str:
    if arm == "trained_ball":
        return "FewShot-R18-bola (treinado)"
    return "FewShot-R18 jogadores (congelado)"


def render_report(results: dict[str, Any]) -> str:
    protocol = results["protocol"]
    splits = results["splits"]
    base = results["baseline_eval"]
    configs = results["configurations"]
    selected_id = results["selected_config_id"]
    selected = next(c for c in configs if c["config_id"] == selected_id)
    frozen_id = results.get("selected_frozen_config_id")
    trained_id = results.get("selected_trained_config_id")
    frozen_sel = next((c for c in configs if c["config_id"] == frozen_id), selected)
    trained_sel = next((c for c in configs if c["config_id"] == trained_id), None) if trained_id else None
    trained_summary = results.get("trained_ball_summary")
    arm3_vs_arm2 = results.get("arm3_vs_arm2_bootstrap")
    headroom = base["headroom"]

    focus = trained_sel if trained_sel is not None else selected
    focus_eval = focus["eval_metrics"]
    focus_cost = focus["filter_cost"]
    focus_boot = focus["bootstrap"]["delta_f1"]
    focus_sep = focus["separability"]

    group_audit = splits.get("group_audit")
    val_status_note = splits.get("val_status_note")
    if group_audit and group_audit.get("train_val"):
        tv = group_audit["train_val"]
        ce = group_audit.get("calib_eval") or {}
        split_bias_detail = (
            f"jogos/grupos: train={tv['groups_a']}, val={tv['groups_b']}, interseção={tv['shared_groups']} "
            f"({'por jogo' if tv['disjoint_by_group'] else 'compartilha jogos'}); "
            f"val dividido por grupo ({ce.get('groups_a', '?')}/{ce.get('groups_b', '?')}, "
            f"interseção={ce.get('shared_groups', '?')})"
        )
    else:
        split_bias_detail = f"partição 50/50 de val por {splits.get('partition_unit', 'image')} (sem campo de jogo)"

    bias_prefix = (
        val_status_note
        or "val.txt não é teste virgem (usado na escolha do checkpoint do detector; comparação relativa justa sob o mesmo viés, números absolutos potencialmente otimistas)"
    )

    if trained_sel is not None:
        frz_boot = frozen_sel["bootstrap"]["delta_f1"]
        line1 = (
            f"1. Veredito (3 braços em `{splits['eval_split_name']}`, n={splits['eval_images']} imagens): "
            f"FewShot-R18-bola treinado ({trained_sel['config_id']}) deu ΔF1={_fmt_signed(focus_boot['point_estimate'])} "
            f"({_direction_pt(focus_boot['direction'])}); controle jogadores congelado ({frozen_sel['config_id']}) deu "
            f"ΔF1={_fmt_signed(frz_boot['point_estimate'])} ({_direction_pt(frz_boot['direction'])})."
        )
    else:
        if focus["calibration"]["is_noop"]:
            line1 = (
                f"1. Veredito: o filtro congelado ({focus['config_id']}) não aplicou corte (limiar -inf na calibração); "
                f"nenhuma detecção foi removida no conjunto final."
            )
        else:
            line1 = (
                f"1. Veredito: em `{splits['eval_split_name']}` (n={splits['eval_images']} imagens), "
                f"a configuração congelada ({focus['config_id']}) variou o F1 em {_fmt_signed(focus_boot['point_estimate'])} "
                f"({_direction_pt(focus_boot['direction'])})."
            )

    line2 = f"2. Viés do split: {bias_prefix}; {split_bias_detail}."

    if trained_sel is not None:
        fe = frozen_sel["eval_metrics"]
        line3 = (
            f"3. Métricas no teste final: Base P={_fmt(base['precision'])}/R={_fmt(base['recall'])}/F1={_fmt(base['f1'])}/mAP50={_fmt(base['map50'])} | "
            f"Jogadores congelado P={_fmt(fe['precision'])}/R={_fmt(fe['recall'])}/F1={_fmt(fe['f1'])}/mAP50={_fmt(fe['map50'])} | "
            f"Bola treinado P={_fmt(focus_eval['precision'])}/R={_fmt(focus_eval['recall'])}/F1={_fmt(focus_eval['f1'])}/mAP50={_fmt(focus_eval['map50'])}."
        )
    else:
        line3 = (
            f"3. Métricas ({protocol['detector_weights']}, imgsz={protocol['imgsz']}, conf={protocol['conf']}): "
            f"Base P={_fmt(base['precision'])}/R={_fmt(base['recall'])}/F1={_fmt(base['f1'])}/mAP50={_fmt(base['map50'])} | "
            f"Filtrado P={_fmt(focus_eval['precision'])}/R={_fmt(focus_eval['recall'])}/F1={_fmt(focus_eval['f1'])}/mAP50={_fmt(focus_eval['map50'])}."
        )

    line4 = (
        f"4. Custo real ({focus['config_id']}): removeu {focus_cost['fps_removed']}/{focus_cost['base_fp']} FPs "
        f"({_fmt(focus_cost['fp_removal_rate'] * 100, 1)}%) e removeu {focus_cost['tps_removed']}/{focus_cost['base_tp']} "
        f"bolas verdadeiras por engano (ΔRecall={_fmt_signed(focus_eval['recall'] - base['recall'])})."
    )

    cluster_boot = focus.get("cluster_bootstrap")
    boot_parts = [
        f"imagens IC 95% [{_fmt_signed(focus_boot['ci_95_low'])}, {_fmt_signed(focus_boot['ci_95_high'])}] ({_direction_short_pt(focus_boot['direction'])})"
    ]
    if cluster_boot is not None:
        cbf1 = cluster_boot["delta_f1"]
        boot_parts.append(
            f"jogos (n={cluster_boot['n_groups']}) IC 95% [{_fmt_signed(cbf1['ci_95_low'])}, {_fmt_signed(cbf1['ci_95_high'])}] ({_direction_short_pt(cbf1['direction'])})"
        )
    if arm3_vs_arm2 is not None:
        a32 = arm3_vs_arm2["delta_f1"]
        boot_parts.append(
            f"treinado−congelado ΔF1={_fmt_signed(a32['point_estimate'])} [{_fmt_signed(a32['ci_95_low'])}, {_fmt_signed(a32['ci_95_high'])}]"
        )
    line5 = f"5. Bootstrap pareado ({protocol['bootstrap_resamples']} reamostragens): " + "; ".join(boot_parts) + "."

    if trained_sel is not None:
        frz_sep = frozen_sel["separability"]
        line6 = (
            f"6. Separabilidade bola vs FP (sem limiar, no eval): "
            f"jogadores congelado ROC-AUC={_fmt(frz_sep['eval_roc_auc'])} (PR-AUC={_fmt(frz_sep['eval_pr_auc'])}); "
            f"bola treinado ROC-AUC={_fmt(focus_sep['eval_roc_auc'])} (PR-AUC={_fmt(focus_sep['eval_pr_auc'])})."
        )
    else:
        line6 = (
            f"6. Separabilidade FewShot-R18 (bola vs FP, sem limiar): "
            f"ROC-AUC={_fmt(focus_sep['eval_roc_auc'])} no eval ({_fmt(focus_sep['calib_roc_auc'])} na calibração); "
            f"PR-AUC={_fmt(focus_sep['eval_pr_auc'])} no eval."
        )

    extra_datasets = results.get("extra_datasets") or []
    extra_parts: list[str] = []
    if trained_summary is not None:
        ps = trained_summary["pool_stats"]
        extra_parts.append(
            f"taxa de FP em train={ps['train_fp_per_image_at_eval_conf']:.3f}/img vs val={base['fp_per_image']:.3f}/img"
        )
    if extra_datasets:
        first_extra = extra_datasets[0]
        eb = first_extra["bootstrap"]["delta_f1"]
        extra_parts.append(
            f"teste virgem `{first_extra['name']}` (n={first_extra['baseline']['n_images']}): "
            f"ΔF1={_fmt_signed(eb['point_estimate'])} [{_fmt_signed(eb['ci_95_low'])}, {_fmt_signed(eb['ci_95_high'])}], "
            f"ROC-AUC={_fmt(first_extra['separability']['roc_auc'])}"
        )
    has_b2b = any("box2box" in d["name"].lower() for d in extra_datasets)
    if not has_b2b:
        extra_parts.append("box2boxcam não medido (inacessível localmente)")
    line7 = "7. Viés de treino e teste virgem: " + "; ".join(extra_parts) + "."

    line8 = (
        f"8. Teto de ganho possível do detector sozinho: {base['fp']} FPs em {base['n_images']} imagens "
        f"({base['fp_per_image']:.3f} FP/img), teto de precisão={_fmt_signed(headroom['precision_headroom'])} "
        f"e teto de F1 sem novo recall={_fmt_signed(headroom['f1_headroom'])}; "
        f"configurações testadas={protocol['total_configs_tested']}."
    )

    lines: list[str] = [
        "# Relatório: Detector de bola + filtro de falsos positivos FewShot-R18",
        "",
        "## Resumo (8 linhas)",
        "",
        line1,
        line2,
        line3,
        line4,
        line5,
        line6,
        line7,
        line8,
        "",
        "## Comparação principal no conjunto de teste final",
        "",
        (
            f"Detector: `{protocol['detector_weights']}` | Controle congelado: `{protocol['fewshot_weights']}` | "
            f"imgsz={protocol['imgsz']} | conf={protocol['conf']} | IoU={protocol['eval_iou']} | "
            f"Split final: `{splits['eval_split_name']}` ({splits['eval_images']} imagens, {splits['eval_gt_balls']} bolas GT)."
        ),
        "",
        "| Braço | Configuração | K | Limiar congelado | Precisão | Recall | F1 | mAP50 | mAP50-95 | FPs removidos | TPs removidos | ΔF1 (IC 95% imgs) | Distinguível de zero? |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        (
            f"| Detector sozinho | baseline | - | - | {_fmt(base['precision'])} | "
            f"{_fmt(base['recall'])} | {_fmt(base['f1'])} | {_fmt(base['map50'])} | "
            f"{_fmt(base['map50_95'])} | 0 / {base['fp']} | 0 / {base['tp']} | 0.0000 | - |"
        ),
    ]

    for cfg in configs:
        em = cfg["eval_metrics"]
        cost = cfg["filter_cost"]
        bf1 = cfg["bootstrap"]["delta_f1"]
        markers = []
        if cfg["config_id"] == frozen_id:
            markers.append("ctrl")
        if cfg["config_id"] == trained_id:
            markers.append("bola")
        if cfg["config_id"] == selected_id:
            markers.append("melhor-calib")
        tag = f" ({', '.join(markers)})" if markers else ""
        dist_label = "Sim" if bf1["distinguishable_from_zero"] else "Não"
        lines.append(
            f"| {_arm_label(cfg.get('arm', 'frozen_players'))} | `{cfg['config_id']}`{tag} | {cfg['k_pos_used']} | "
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
            "Seleção de checkpoint, camada, K e limiar ocorreu exclusivamente em `val_calib`, congelada antes de medir `val_eval`.",
            "",
            "## O FewShot-R18 separa bola de falso positivo?",
            "",
            "| Braço | Configuração | Camada | Modo | K pos / neg | ROC-AUC (calib) | PR-AUC (calib) | ROC-AUC (eval) | PR-AUC (eval) | TP / FP (eval) |",
            "|---|---|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )

    det_sep = results.get("detector_conf_separability")
    if det_sep is not None:
        lines.append(
            f"| Detector sozinho (referência `conf`) | `detector_conf` | - | `conf` | - | "
            f"{_fmt(det_sep['calib_roc_auc'])} | {_fmt(det_sep['calib_pr_auc'])} | "
            f"{_fmt(det_sep['eval_roc_auc'])} | {_fmt(det_sep['eval_pr_auc'])} | {base['tp']} / {base['fp']} |"
        )
    for cfg in configs:
        sep = cfg["separability"]
        lines.append(
            f"| {_arm_label(cfg.get('arm', 'frozen_players'))} | `{cfg['config_id']}` | `{cfg['layer']}` | `{cfg['mode']}` | "
            f"{cfg['k_pos_used']} / {cfg['k_neg_used']} | {_fmt(sep['calib_roc_auc'])} | "
            f"{_fmt(sep['calib_pr_auc'])} | {_fmt(sep['eval_roc_auc'])} | "
            f"{_fmt(sep['eval_pr_auc'])} | {sep['eval_n_tp']} / {sep['eval_n_fp']} |"
        )

    if trained_summary is not None:
        ps = trained_summary["pool_stats"]
        lines.extend(
            [
                "",
                "## Treino episódico do FewShot-R18-bola (receita do paper)",
                "",
                f"- Arquitetura e otimização: ResNet-18 ImageNet (`fc=Identity`, apenas `layer3` e `layer4` treináveis, `BatchNorm2d` em `eval`), entrada letterbox RGB `{trained_summary['canvas_wh']}`, `AdamW(lr={trained_summary['lr']}, wd={trained_summary['weight_decay']})`, `CosineAnnealingLR`, `{trained_summary['epochs']}` épocas × `{trained_summary['steps_per_epoch']}` episódios (`k_suporte={trained_summary['references_per_class']}`, `n_query={trained_summary['queries_per_class']}` por classe, `cosine * {trained_summary['cosine_scale']}`, seed `{trained_summary['seed']}`).",
                f"- Dispositivo e velocidade: `{trained_summary['device']}`, 1º episódio `{trained_summary['first_episode_s']}s`, tempo total `{trained_summary['total_train_s']}s`, melhor época selecionada em `val_calib`: época `{trained_summary['best_epoch']}`.",
                f"- Mineração em `train.txt` (`{ps['train_images_scanned']}` imagens varridas com `conf>={ps['mining_conf']}`): `{ps['pool_pos_crops']}` recortes positivos, `{ps['train_fp_high_conf_mined']}` FPs de alta confiança (`>={ps['eval_conf']}`), `{ps['train_fp_low_conf_mined']}` FPs minerados em baixa confiança (`{ps['mining_conf']}..{ps['eval_conf']}`), `{ps['background_supplement_crops']}` recortes suplementares de fundo.",
                f"- Viés de treino do detector (`conf>={ps['eval_conf']}`): taxa de FP em `train` = `{ps['train_fp_per_image_at_eval_conf']:.4f}` FP/img (`{ps['train_fp_at_eval_conf']}` FPs, precisão `{ps['train_precision_at_eval_conf']:.4f}`) vs em `{splits['eval_split_name']}` = `{base['fp_per_image']:.4f}` FP/img (`{base['fp']}` FPs, precisão `{base['precision']:.4f}`).",
                "",
                "| Época | Loss treino | ROC-AUC (`val_calib`) | PR-AUC (`val_calib`) | Melhor F1 (`val_calib`) | Tempo (s) |",
                "|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in trained_summary["history"]:
            mark = " *" if row["epoch"] == trained_summary["best_epoch"] else ""
            lines.append(
                f"| {row['epoch']}{mark} | {row['loss']:.4f} | {_fmt(row['val_calib_roc_auc'])} | "
                f"{_fmt(row['val_calib_pr_auc'])} | {_fmt(row['val_calib_best_f1'])} | {row['elapsed_s']:.1f} |"
            )

    lines.extend(
        [
            "",
            "## Teto de ganho possível (headroom)",
            "",
            f"- Detecções base no eval: {base['total_pred']} ({base['tp']} TP, {base['fp']} FP, {base['fn']} FN; {base['fp_per_image']:.3f} FP/imagem).",
            f"- Precisão base: {_fmt(base['precision'])} → teto teórico com filtro perfeito: {_fmt(headroom['max_precision_perfect_filter'])} (ganho máximo de precisão: {_fmt_signed(headroom['precision_headroom'])}).",
            f"- F1 base: {_fmt(base['f1'])} → teto teórico sem novo recall: {_fmt(headroom['max_f1_perfect_filter'])} (ganho máximo de F1: {_fmt_signed(headroom['f1_headroom'])}).",
            "",
            "## Protocolo, higiene de splits e escolhas",
            "",
            f"- Fonte dos splits: `{protocol['data_yaml']}` (lista de validação: `{splits.get('val_list_source', 'data.yaml:val')}`; disjunção `train`/`val` verificada: `{splits['train_val_disjoint_verified']}`).",
            f"- Referências do filtro: sorteadas automaticamente apenas de `train` (`train_ref`: {splits['train_ref_images']} imagens, {splits['train_ref_gt_balls']} bolas GT) com seed fixa `{protocol['seed']}`.",
            f"- Calibração do limiar: `{protocol['calib_source']}` (`{splits['calib_split_name']}`, {splits['calib_images']} imagens, fração `{protocol['calib_fraction']}`). Limiar congelado antes de tocar `{splits['eval_split_name']}` ({splits['eval_images']} imagens).",
            f"- Pré-processamento do crop: canvas letterbox RGB `{protocol['canvas_wh']}`, `context_scale={protocol['context_scale']}`, `square_crop={protocol['square_crop']}`.",
            f"- Configurações testadas: {protocol['total_configs_tested']} (política de variações `{protocol['variations_policy']}`, gatilho AUC `< {protocol['auc_variation_trigger']}`, variações executadas: `{protocol['variations_executed']}`).",
            f"- Escolha geral na calibração: `{selected_id}` — {results['selection_rationale']}.",
        ]
    )

    if group_audit and group_audit.get("train_val"):
        tv = group_audit["train_val"]
        ce = group_audit.get("calib_eval") or {}
        lines.append(
            f"- Auditoria por jogo/grupo: `train`={tv['groups_a']} grupos, `val`={tv['groups_b']} grupos, "
            f"interseção `train∩val`={tv['shared_groups']} ({'split por jogo' if tv['disjoint_by_group'] else 'compartilha jogos entre train e val'}); "
            f"`{splits['calib_split_name']}`={ce.get('groups_a', '?')} grupos vs `{splits['eval_split_name']}`={ce.get('groups_b', '?')} grupos "
            f"(interseção={ce.get('shared_groups', '?')})."
        )
    lines.append(f"- Status do conjunto de validação: {bias_prefix}.")

    val_sampling = splits.get("val_sampling") or {}
    if val_sampling.get("subsampled"):
        lines.append(
            f"- Amostragem estratificada de `val`: {val_sampling['sampled_images']}/{val_sampling['original_images']} "
            f"imagens (seed `{val_sampling['seed']}`, {val_sampling.get('n_strata', '?')} estratos)."
        )

    if protocol["skipped_variants"]:
        lines.append("")
        lines.append("Variações não executadas:")
        for skipped in protocol["skipped_variants"]:
            lines.append(f"- `{skipped['variant']}`: {skipped['reason']}")

    lines.extend(["", "## Conjuntos extras / teste virgem fora de r1", ""])
    if not extra_datasets:
        lines.append(
            "Não medido: nenhum conjunto extra (como `box2boxcam` ou dataset virgem fora de `r1`) foi fornecido nesta execução."
        )
    else:
        for ds in extra_datasets:
            db = ds["baseline"]
            lines.extend(
                [
                    f"### {ds['name']}",
                    "",
                    f"Imagens: {db['n_images']} | Bolas GT: {db['total_gt']} | Protótipos e limiares congelados (sem reajuste neste conjunto).",
                    "",
                    "| Braço | Configuração congelada | Limiar | Precisão | Recall | F1 | mAP50 | FPs removidos | TPs removidos | ROC-AUC | ΔF1 (IC 95%) |",
                    "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
                    (
                        f"| Detector sozinho | baseline | - | {_fmt(db['precision'])} | {_fmt(db['recall'])} | "
                        f"{_fmt(db['f1'])} | {_fmt(db['map50'])} | 0 / {db['fp']} | 0 / {db['tp']} | "
                        f"{_fmt((ds.get('detector_conf_separability') or {}).get('roc_auc'))} | 0.0000 |"
                    ),
                ]
            )
            arms_map = ds.get("arms") or {"selected": ds}
            for arm_key, arm_res in arms_map.items():
                af = arm_res["filtered"]
                ac = arm_res["filter_cost"]
                ab = arm_res["bootstrap"]["delta_f1"]
                asep = arm_res["separability"]
                lines.append(
                    f"| {_arm_label(arm_key)} | `{arm_res['applied_config_id']}` | {_fmt_threshold(arm_res['frozen_threshold'])} | "
                    f"{_fmt(af['precision'])} | {_fmt(af['recall'])} | {_fmt(af['f1'])} | {_fmt(af['map50'])} | "
                    f"{ac['fps_removed']} / {ac['base_fp']} | {ac['tps_removed']} / {ac['base_tp']} | "
                    f"{_fmt(asep['roc_auc'])} | {_fmt_signed(ab['point_estimate'])} [{_fmt_signed(ab['ci_95_low'])}, {_fmt_signed(ab['ci_95_high'])}] |"
                )
            lines.append("")
        if not has_b2b:
            lines.append("- `box2boxcam` (câmera amadora/noturna, 198 quadros): não medido por não estar disponível localmente.")

    lines.extend(["", "## Limitações", ""])
    limitations = [
        "O conjunto `val.txt` do `r5_9010` não é teste virgem do detector: foi usado durante o treino do `t1b_v5_etapa1_rodada1.pt` para selecionar o melhor checkpoint.",
        "O filtro atua apenas em pós-processamento (remoção de FPs) e nunca aumenta recall nem recupera bolas não propostas pelo detector.",
        "Bolas em quadros 1080p frequentemente medem ~10×10 pixels; o redimensionamento para 48×96 letterbox pode perder detalhes finos e depende fortemente da aparência local.",
        "A mineração de negativos em `train.txt` sofre do viés de treino do detector (menos FPs de alta confiança do que fora do treino), exigindo mineração em confiança baixa (`0.05..0.35`).",
        "O conjunto amador noturno `box2boxcam` (198 quadros) não estava disponível localmente para medição direta.",
    ]
    for warning in results.get("warnings") or []:
        limitations.append(f"Aviso desta execução: {warning}.")
    for item in limitations:
        lines.append(f"- {item}")

    lines.append("")
    return "\n".join(lines)
