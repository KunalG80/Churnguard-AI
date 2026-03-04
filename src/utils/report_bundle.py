def build_report_bundle(
    seg_df,
    summary,
    retention_cost,
    months_lost,
    success_rate,
    total_budget
):

    bundle = {
        "segmentation": seg_df,
        "summary": summary,
        "retention_cost": retention_cost,
        "months_lost": months_lost,
        "success_rate": success_rate,
        "budget": total_budget
    }

    return bundle