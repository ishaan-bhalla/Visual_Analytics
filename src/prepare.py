"""Clean and merge the 2011 and 2021 Census tables into one area-level dataset.

Every step logs how many areas it keeps, so the README can show exactly where
areas drop out (406 UK areas -> 348 in England & Wales -> 323 matched in 2021).

Run from the repo root:  python src/prepare.py
"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
GEO = ROOT / "data" / "geo"
OUT = ROOT / "data" / "processed"

LOG: dict = {"steps": [], "missing_values": {}, "recoded_areas": [], "unmatched_2011_areas": []}

# The Census tables use the codes ONS reissued after small boundary changes in
# 2013-14; the December 2011 boundary file and region lookup use the originals.
# Map the newer code to the original so every area joins to its 2011 polygon.
RECODED_TO_2011 = {
    "E06000057": "E06000048",  # Northumberland (2013)
    "E08000037": "E08000020",  # Gateshead (2013)
    "E07000240": "E07000100",  # St Albans (2014)
    "E07000241": "E07000104",  # Welwyn Hatfield (2014)
    "E07000242": "E07000097",  # East Hertfordshire (2014)
    "E07000243": "E07000101",  # Stevenage (2014)
}


def log_step(name: str, rows: int, note: str = "") -> None:
    LOG["steps"].append({"step": name, "areas": rows, "note": note})
    print(f"{name:<45} {rows:>4} areas  {note}")


def load(file: str, rename: dict) -> pd.DataFrame:
    df = pd.read_csv(RAW / file)
    df = df.rename(columns={"geography": "area", "geography code": "area_code", **rename})
    df["area"] = df["area"].astype(str).str.strip()
    df["area_code"] = df["area_code"].astype(str).str.strip().str.upper()
    dupes = int(df["area_code"].duplicated().sum())
    assert dupes == 0, f"{file}: {dupes} duplicate area codes"
    return df


def pct(df: pd.DataFrame, num: str, den: str) -> pd.Series:
    return df[num] / df[den] * 100


def employment() -> pd.DataFrame:
    p = "Economic Activity: "
    s = "; measures: Value"
    df = load("employment.csv", {
        f"{p}All usual residents aged 16 to 74{s}": "residents_16_74",
        f"{p}Economically active{s}": "active",
        f"{p}Economically active: In employment{s}": "employed",
        f"{p}Economically active: Employee: Part-time{s}": "part_time",
        f"{p}Economically active: Employee: Full-time{s}": "full_time",
        f"{p}Economically active: Unemployed{s}": "unemployed",
        f"{p}Economically Inactive{s}": "inactive",
    })
    # Employment and unemployment are shares of the economically active
    # population; they don't sum to 100 because full-time students who are
    # active are a separate category. Inactivity is a share of all 16-74s.
    df["employment_pct"] = pct(df, "employed", "active")
    df["unemployment_pct"] = pct(df, "unemployed", "active")
    df["inactivity_pct"] = pct(df, "inactive", "residents_16_74")
    df["full_time_pct"] = pct(df, "full_time", "employed")
    df["part_time_pct"] = pct(df, "part_time", "employed")
    return df[["area", "area_code", "employment_pct", "unemployment_pct", "inactivity_pct",
               "full_time_pct", "part_time_pct"]]


def ethnicity() -> pd.DataFrame:
    p = "Ethnic Group: "
    s = "; measures: Value"
    df = load("ethnic_groups.csv", {
        f"{p}All usual residents{s}": "residents",
        f"{p}White{s}": "white",
        f"{p}Mixed/multiple ethnic groups{s}": "mixed",
        f"{p}Asian/Asian British{s}": "asian",
        f"{p}Black/African/Caribbean/Black British{s}": "black",
        f"{p}Other ethnic group{s}": "other",
    })
    for g in ["white", "mixed", "asian", "black", "other"]:
        df[f"{g}_pct"] = pct(df, g, "residents")
    return df[["area", "area_code", "white_pct", "mixed_pct", "asian_pct", "black_pct", "other_pct"]]


def migration() -> pd.DataFrame:
    p = "Migration/ethnic group: "
    s = "; measures: Value"
    df = load("migration.csv", {
        f"{p}All usual residents{s}": "residents",
        f"{p}Lived at same address one year ago{s}": "same_address",
        f"{p}Migrants: Moved within the area{s}": "moved_within_area",
        f"{p}Migrants: Moved into the area from within the UK{s}": "moved_from_uk",
        f"{p}Migrants: Moved into the area from outside the UK{s}": "moved_from_outside_uk",
        f"{p}Moved out of the area{s}": "moved_out",
    })
    for c in ["same_address", "moved_within_area", "moved_from_uk", "moved_from_outside_uk", "moved_out"]:
        df[f"{c}_pct"] = pct(df, c, "residents")
    return df[["area", "area_code", "same_address_pct", "moved_within_area_pct", "moved_from_uk_pct",
               "moved_from_outside_uk_pct", "moved_out_pct"]]


def health_2011() -> pd.DataFrame:
    p = "disability/health/care: "
    s = "; measures: Value"
    df = load("health.csv", {
        f"{p}All categories: Long-term health problem or disability{s}": "residents",
        f"{p}Day-to-day activities limited a lot{s}": "limited_a_lot",
        f"{p}Day-to-day activities limited a little{s}": "limited_a_little",
        f"{p}Day-to-day activities not limited{s}": "not_limited",
        f"{p}Very good health{s}": "very_good",
        f"{p}Good health{s}": "good",
        f"{p}Fair health{s}": "fair",
        f"{p}Bad health{s}": "bad",
        f"{p}Very bad health{s}": "very_bad",
    })
    df["limited_any_pct"] = pct(df, "limited_a_lot", "residents") + pct(df, "limited_a_little", "residents")
    df["not_limited_pct"] = pct(df, "not_limited", "residents")
    df["good_or_very_good_health_pct"] = pct(df, "very_good", "residents") + pct(df, "good", "residents")
    df["fair_health_pct"] = pct(df, "fair", "residents")
    df["bad_or_very_bad_health_pct"] = pct(df, "bad", "residents") + pct(df, "very_bad", "residents")
    return df[["area", "area_code", "limited_any_pct", "not_limited_pct", "good_or_very_good_health_pct",
               "fair_health_pct", "bad_or_very_bad_health_pct"]]


def health_2021() -> pd.DataFrame:
    p = "General health: "
    df = load("health21.csv", {
        f"{p}Total: All usual residents": "residents",
        f"{p}Very good health": "very_good",
        f"{p}Good health": "good",
        f"{p}Fair health": "fair",
        f"{p}Bad health": "bad",
        f"{p}Very bad health": "very_bad",
    })
    df["good_or_very_good_health_pct"] = pct(df, "very_good", "residents") + pct(df, "good", "residents")
    df["fair_health_pct"] = pct(df, "fair", "residents")
    df["bad_or_very_bad_health_pct"] = pct(df, "bad", "residents") + pct(df, "very_bad", "residents")
    return df[["area", "area_code", "good_or_very_good_health_pct", "fair_health_pct",
               "bad_or_very_bad_health_pct"]]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    emp, eth, mig, h11, h21 = employment(), ethnicity(), migration(), health_2011(), health_2021()
    log_step("Employment 2011 (KS601UK, UK)", len(emp), "UK-wide table")
    log_step("Migration 2011 (UK)", len(mig), "UK-wide table")
    log_step("Ethnic group 2011 (KS201EW)", len(eth), "England & Wales only")
    log_step("Health 2011 (KS301EW)", len(h11), "England & Wales only")
    log_step("General health 2021 (TS037)", len(h21), "2021 council boundaries")

    # Merge on the area code only; names are kept from the health table.
    base = h11[["area", "area_code"]]
    df = base
    for part in (emp, eth, mig, h11):
        df = df.merge(part.drop(columns="area"), on="area_code", how="inner")
    log_step("2011 tables merged (inner join on code)", len(df),
             "keeps England & Wales; drops Scotland and Northern Ireland")

    for name, part in {"employment": emp, "ethnicity": eth, "migration": mig,
                       "health_2011": h11, "health_2021": h21}.items():
        LOG["missing_values"][name] = int(part.isna().sum().sum())

    # Attach region and confirm every area has a 2011 boundary for the map.
    df["geo_code"] = df["area_code"].replace(RECODED_TO_2011)
    LOG["recoded_areas"] = [{"census_code": k, "boundary_2011_code": v} for k, v in RECODED_TO_2011.items()]
    regions = pd.read_csv(GEO / "lad11_to_region11.csv")
    df = df.merge(regions[["lad11cd", "rgn11nm"]].rename(columns={"lad11cd": "geo_code", "rgn11nm": "region"}),
                  on="geo_code", how="left")
    assert df["region"].notna().all(), "area without a region"
    geo_codes = {f["properties"]["lad11cd"] for f in json.loads((GEO / "lad_dec2011_ew.geojson").read_text())["features"]}
    assert set(df["geo_code"]) <= geo_codes, "area without a 2011 boundary"

    # 2011 -> 2021: match on code. A 2011 code missing in 2021 means the council
    # was merged into a new one between 2019 and 2023, so its health can't be
    # compared like for like.
    h21r = h21.rename(columns={c: f"{c}_2021" for c in h21.columns if c.endswith("_pct")})
    df = df.merge(h21r.drop(columns="area"), on="area_code", how="left")
    df["good_change"] = df["good_or_very_good_health_pct_2021"] - df["good_or_very_good_health_pct"]
    df["fair_change"] = df["fair_health_pct_2021"] - df["fair_health_pct"]
    df["bad_change"] = df["bad_or_very_bad_health_pct_2021"] - df["bad_or_very_bad_health_pct"]
    df["matched_2021"] = df["good_change"].notna()
    unmatched = df.loc[~df["matched_2021"], ["area", "area_code", "region"]]
    LOG["unmatched_2011_areas"] = unmatched.to_dict("records")
    log_step("Matched to 2021 by area code", int(df["matched_2021"].sum()),
             f"{len(unmatched)} areas merged into new councils 2019-2023")

    df = df.round(4)
    df.to_csv(OUT / "areas_2011_2021.csv", index=False)
    (OUT / "pipeline_log.json").write_text(json.dumps(LOG, indent=2))
    print("\nUnmatched 2011 areas:")
    print(unmatched.to_string(index=False))


if __name__ == "__main__":
    main()
