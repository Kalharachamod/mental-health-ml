def what_if_rows(base_row: dict, scenarios: dict):
    rows = []
    for name, chg in scenarios.items():
        r = dict(base_row)
        r.update(chg)
        rows.append((name, r))
    return rows
