def lifestyle_recommendations(row: dict):
    tips=[]
    sleep=float(row.get("Sleep_Hours",7) or 7)
    work=float(row.get("Work_Hours",40) or 40)
    act=float(row.get("Physical_Activity_Hours",3) or 3)
    smu=float(row.get("Social_Media_Usage",2) or 2)
    diet=str(row.get("Diet_Quality","")).lower()
    smoke=str(row.get("Smoking_Habit","")).lower()
    alc=str(row.get("Alcohol_Consumption","")).lower()

    if sleep<7: tips.append("Aim for 7–9 hours of sleep; keep a consistent schedule.")
    if work>55: tips.append("Reduce weekly work hours; schedule breaks to manage stress.")
    if act<3: tips.append("Add 30–45 minutes of moderate physical activity most days.")
    if smu>4: tips.append("Reduce social media time; set daily app limits.")
    if diet.startswith("unhealthy"): tips.append("Improve diet quality with more whole foods.")
    if smoke in ["regular smoker","heavy smoker"]: tips.append("Consider a smoking cessation plan.")
    if alc=="regular drinker": tips.append("Limit alcohol to low-risk levels or reduce frequency.")
    return tips
