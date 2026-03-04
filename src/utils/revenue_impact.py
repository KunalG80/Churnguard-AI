def revenue_impact(
    df,
    retention_cost,
    retention_success_rate,
    months_lost,
    threshold=0.5
):

    # BUSINESS TARGETING BASED ON RISK
    target = df[
        df["Churn_Probability"] >= threshold
    ]

    predicted_churners = len(target)

    potential_loss = (
        target["MonthlyCharges"].sum()
        * months_lost
    )

    revenue_saved = (
        potential_loss
        * retention_success_rate
    )

    retention_spend = (
        predicted_churners
        * retention_cost
    )

    net_impact = revenue_saved - retention_spend

    return {
        "predicted_churners": predicted_churners,
        "potential_loss": potential_loss,
        "revenue_saved": revenue_saved,
        "net_impact": net_impact
    }