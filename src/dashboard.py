"""Build the interactive linked dashboard (docs/index.html) and the README figures.

Every view is drawn from the same per-area table, so one selection drives them
all: drag a box on the PCA plot, or click areas on any map or scatter (shift
for several), and the same areas light up everywhere. The region menu filters
every view.

Run from the repo root:  python src/dashboard.py
"""
import json
from pathlib import Path

import altair as alt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"
GEO = ROOT / "data" / "geo" / "lad_dec2011_ew.geojson"
DOCS = ROOT / "docs"
FIGS = DOCS / "figures"

CLUSTER_COLOURS = alt.Scale(domain=["Diverse, high-migration urban areas", "Less diverse, lower-migration areas"],
                            range=["#d95f02", "#1b9e77"])
NO_DATA = "#9e9e9e"
W, H = 290, 320  # map size


def _signed_area(ring: list) -> float:
    return sum(x1 * y2 - x2 * y1 for (x1, y1), (x2, y2) in zip(ring, ring[1:] + ring[:1])) / 2


def rewind(feature: dict) -> dict:
    """ONS GeoJSON follows RFC 7946 (anticlockwise outer rings); Vega's d3
    projection expects the opposite and would otherwise fill the whole globe."""
    def fix(poly: list) -> list:
        return [r if (_signed_area(r) < 0) == (i == 0) else r[::-1] for i, r in enumerate(poly)]
    g = feature["geometry"]
    coords = [fix(p) for p in g["coordinates"]] if g["type"] == "MultiPolygon" else fix(g["coordinates"])
    return {**feature, "geometry": {"type": g["type"], "coordinates": coords}}


def load() -> tuple[pd.DataFrame, dict, list]:
    df = pd.read_csv(DATA / "analysis.csv")
    summary = json.loads((ROOT / "results" / "summary.json").read_text())
    geo = [rewind(f) for f in json.loads(GEO.read_text())["features"]]
    df["status"] = df["matched_2021"].map({True: "Matched to 2021", False: "Council merged 2019-21 (no 2021 match)"})
    return df, summary, geo


def build(df: pd.DataFrame, summary: dict, geo: list) -> tuple[alt.VConcatChart, dict[str, alt.Chart]]:
    regions = sorted(df.region.unique())
    region = alt.param(name="region", value="All",
                       bind=alt.binding_select(options=["All"] + regions, name="Filter by region: ",
                                                 element="#controls"))
    pick = alt.selection_point(name="pick", fields=["area_code"], on="click", toggle="event.shiftKey",
                               clear="dblclick")
    brush = alt.selection_interval(name="brush", encodings=["x", "y"], clear="dblclick")
    region_filter = "region == 'All' || datum.region == region"
    # An area is "active" if it matches both the click selection and the brush.
    active = pick & brush

    base = alt.Chart(df).transform_filter(region_filter)
    tooltip = [alt.Tooltip("area:N", title="Area"), alt.Tooltip("region:N", title="Region"),
               alt.Tooltip("cluster:N", title="Cluster"),
               alt.Tooltip("good_or_very_good_health_pct:Q", title="Good health 2011 (%)", format=".1f"),
               alt.Tooltip("good_or_very_good_health_pct_2021:Q", title="Good health 2021 (%)", format=".1f"),
               alt.Tooltip("good_change:Q", title="Change (pp)", format="+.2f"),
               alt.Tooltip("residual:Q", title="Model residual (pp)", format="+.2f"),
               alt.Tooltip("unemployment_pct:Q", title="Unemployment (%)", format=".1f"),
               alt.Tooltip("status:N", title="2021 match")]

    def geomap(field: str, title: str, scheme: str, domain_mid: float | None, fmt: str, legend: str) -> alt.Chart:
        scale = alt.Scale(scheme=scheme, domainMid=domain_mid, reverse=False) if domain_mid is not None \
            else alt.Scale(scheme=scheme)
        shapes = base.transform_lookup(lookup="geo_code",
                                       from_=alt.LookupData(alt.NamedData(name="lad11"), key="properties.lad11cd"),
                                       as_="geo")
        # Vega-Lite drops shapes whose colour value is null, so draw every area
        # in grey first; areas with no value (merged councils) stay grey.
        under = shapes.mark_geoshape(fill=NO_DATA, stroke="white", strokeWidth=0.25).encode(
            shape="geo:G", tooltip=tooltip)
        over = (shapes.transform_filter(f"isValid(datum.{field})")
                .mark_geoshape(stroke="white", strokeWidth=0.25)
                .encode(shape="geo:G",
                        color=alt.Color(f"{field}:Q", scale=scale,
                                        legend=alt.Legend(title=legend, orient="bottom", format=fmt,
                                                          gradientLength=200)),
                        opacity=alt.condition(active, alt.value(1), alt.value(0.25)),
                        tooltip=tooltip)
                .add_params(pick))
        return (under + over).project(type="mercator").properties(width=W, height=H, title=title)

    map_2011 = geomap("good_or_very_good_health_pct", "Good health, 2011", "greens", None, ".0f", "% good/very good")
    map_change = geomap("good_change", "Change in good health, 2011 to 2021", "redblue", 0, "+.1f",
                        "pp (grey: merged council)")
    map_resid = geomap("residual", "Where the model is wrong (residuals)", "purpleorange", 0, "+.1f",
                       "observed minus predicted (pp)")

    pc1 = summary["pca_socio_15"]["explained_variance"][0] * 100
    pc2 = summary["pca_socio_15"]["explained_variance"][1] * 100
    pca = (base.mark_circle(size=40, stroke="black", strokeWidth=0.3)
           .encode(x=alt.X("pc1:Q", title=f"PC1 ({pc1:.0f}%): more ethnically diverse, more migration →"),
                   y=alt.Y("pc2:Q", title=f"PC2 ({pc2:.0f}%): higher unemployment and inactivity →"),
                   color=alt.Color("cluster:N", scale=CLUSTER_COLOURS,
                                   legend=None),
                   opacity=alt.condition(active, alt.value(0.9), alt.value(0.1)),
                   tooltip=tooltip)
           .add_params(brush, pick)
           .properties(width=W, height=280, title=alt.Title(
               "Socio-economic PCA (drag to select)",
               subtitle=["k-means clusters (k=2): orange = diverse, high-migration urban areas;",
                         "green = less diverse, lower-migration areas"])))

    tsne = (base.mark_circle(size=30)
            .encode(x=alt.X("tsne1:Q", title="t-SNE 1", axis=alt.Axis(labels=False, ticks=False)),
                    y=alt.Y("tsne2:Q", title="t-SNE 2", axis=alt.Axis(labels=False, ticks=False)),
                    color=alt.Color("cluster:N", scale=CLUSTER_COLOURS, legend=None),
                    opacity=alt.condition(active, alt.value(0.9), alt.value(0.1)),
                    tooltip=tooltip)
            .add_params(pick)
            .properties(width=W, height=280, title="t-SNE: local neighbourhoods"))

    catchup = (base.transform_filter("datum.matched_2021")
               .mark_circle(size=35)
               .encode(x=alt.X("good_or_very_good_health_pct:Q", title="Good health in 2011 (%)",
                               scale=alt.Scale(zero=False)),
                       y=alt.Y("good_change:Q", title="Change to 2021 (pp)"),
                       color=alt.Color("cluster:N", scale=CLUSTER_COLOURS, legend=None),
                       opacity=alt.condition(active, alt.value(0.9), alt.value(0.1)),
                       tooltip=tooltip)
               .add_params(pick)
               .properties(width=W, height=280, title="Catch-up: less healthy areas improved more"))
    catchup = catchup + catchup.transform_regression("good_or_very_good_health_pct", "good_change") \
        .mark_line(color="black", strokeDash=[4, 3]).encode(color=alt.value("black"), opacity=alt.value(1))

    observed = (base.transform_filter("datum.matched_2021")
                .mark_circle(size=35)
                .encode(x=alt.X("predicted_change:Q", title="Predicted change (pp)"),
                        y=alt.Y("good_change:Q", title="Observed change (pp)"),
                        color=alt.Color("residual:Q", scale=alt.Scale(scheme="purpleorange", domainMid=0),
                                        legend=None),
                        opacity=alt.condition(active, alt.value(0.9), alt.value(0.1)),
                        tooltip=tooltip)
                .add_params(pick)
                .properties(width=W, height=280, title="Model fit: observed vs predicted"))
    lims = [float(df.good_change.min()), float(df.good_change.max())]
    diagonal = alt.Chart(pd.DataFrame({"x": lims, "y": lims})).mark_line(color="grey", strokeDash=[4, 3]) \
        .encode(x="x:Q", y="y:Q")
    observed = observed + diagonal

    coefs = []
    for model, label in [("marked_version", "Marked version"), ("extended", "Extended (adds 2011 health)")]:
        for row in summary["models"][model]["coefficients_per_sd"]:
            if row["term"] != "intercept":
                coefs.append({**row, "model": label})
    names = {"good_or_very_good_health_pct": "Good health 2011", "unemployment_pct": "Unemployment",
             "inactivity_pct": "Inactivity", "asian_pct": "Asian population",
             "moved_from_outside_uk_pct": "Arrived from outside UK"}
    cdf = pd.DataFrame(coefs).assign(term=lambda d: d.term.map(names))
    cbase = alt.Chart(cdf).encode(y=alt.Y("term:N", title=None, sort=list(names.values())),
                                  yOffset=alt.YOffset("model:N"),
                                  color=alt.Color("model:N", scale=alt.Scale(domain=["Marked version", "Extended (adds 2011 health)"],
                                                                  range=["#9e9e9e", "#3f51b5"]),
                                                  legend=alt.Legend(title=None, orient="bottom", columns=1)))
    coef_chart = (cbase.mark_rule(strokeWidth=2).encode(x=alt.X("ci_low:Q", title="Effect per 1 SD (pp), 95% CI"),
                                                        x2="ci_high:Q")
                  + cbase.mark_point(filled=True, size=50).encode(
                      x="mean:Q", tooltip=["model", "term", alt.Tooltip("mean:Q", format="+.2f"),
                                           alt.Tooltip("ci_low:Q", format="+.2f"),
                                           alt.Tooltip("ci_high:Q", format="+.2f")])
                  + alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color="black").encode(x="x:Q")
                  ).properties(width=W, height=280, title="Bayesian regression coefficients")

    region_view = (base.transform_filter("datum.matched_2021")
                   .transform_aggregate(y2011="mean(good_or_very_good_health_pct)",
                                        y2021="mean(good_or_very_good_health_pct_2021)", groupby=["region"]))
    dumbbell = (region_view.mark_rule(color="grey").encode(
                    y=alt.Y("region:N", title=None, sort=alt.EncodingSortField("y2021", order="descending")), x=alt.X("y2011:Q", title="Good health (%)",
                                                                       scale=alt.Scale(zero=False)), x2="y2021:Q")
                + region_view.mark_circle(size=60, color="#bdbdbd").encode(y=alt.Y("region:N", sort=alt.EncodingSortField("y2021", order="descending")), x="y2011:Q",
                                                                            tooltip=["region", alt.Tooltip("y2011:Q", format=".1f", title="2011")])
                + region_view.mark_circle(size=60, color="#238b45").encode(y=alt.Y("region:N", sort=alt.EncodingSortField("y2021", order="descending")), x="y2021:Q",
                                                                            tooltip=["region", alt.Tooltip("y2021:Q", format=".1f", title="2021")])
                ).properties(width=W, height=280, title="Regions: 2011 (grey) to 2021 (green)")

    maps = alt.hconcat(map_2011, map_change, map_resid).resolve_scale(color="independent")
    explore = alt.hconcat(pca, tsne, catchup).resolve_scale(color="shared")
    model = alt.hconcat(observed, coef_chart, dumbbell).resolve_scale(color="independent")
    dash = (alt.vconcat(maps, explore, model)
            .add_params(region)
            .properties(datasets={"lad11": geo})
            .configure_view(stroke=None)
            .configure_title(fontSize=13, anchor="start")
            .configure_axis(labelFontSize=10, titleFontSize=11))
    figures = {"maps": maps, "explore": explore, "model": model}
    return dash, figures


PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Health and Inequality, England & Wales</title>
<link rel="icon" type="image/svg+xml" href="data:image/svg+xml,%3Csvg xmlns=%27http://www.w3.org/2000/svg%27 viewBox=%270 0 32 32%27%3E%3Crect width=%2732%27 height=%2732%27 rx=%277%27 fill=%27%23238b45%27/%3E%3Crect x=%277%27 y=%2717%27 width=%275%27 height=%278%27 rx=%271%27 fill=%27white%27/%3E%3Crect x=%2714%27 y=%2711%27 width=%275%27 height=%2714%27 rx=%271%27 fill=%27white%27/%3E%3Crect x=%2721%27 y=%276%27 width=%275%27 height=%2719%27 rx=%271%27 fill=%27white%27/%3E%3C/svg%3E">
<script src="https://cdn.jsdelivr.net/npm/vega@{vega}"></script>
<script src="https://cdn.jsdelivr.net/npm/vega-lite@{vegalite}"></script>
<script src="https://cdn.jsdelivr.net/npm/vega-embed@{embed}"></script>
<style>
  :root {{ --fg:#1d1d1f; --muted:#5f6368; --bg:#ffffff; --card:#f6f7f9; --accent:#238b45; }}
  body {{ margin:0; background:var(--bg); color:var(--fg);
         font:15px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }}
  main {{ max-width:1180px; margin:0 auto; padding:24px 16px 48px; }}
  h1 {{ font-size:26px; margin:0 0 4px; }}
  .sub {{ color:var(--muted); margin:0 0 18px; }}
  .findings {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(230px,1fr)); gap:10px; margin:0 0 18px; }}
  .card {{ background:var(--card); border-radius:8px; padding:10px 12px; }}
  .card b {{ display:block; font-size:20px; color:var(--accent); }}
  .how {{ color:var(--muted); font-size:13px; margin:0 0 10px; }}
  #vis {{ overflow-x:auto; }}
  #controls {{ margin:0 0 10px; font-size:14px; }}
  #controls select {{ font-size:14px; padding:2px 4px; }}
  footer {{ color:var(--muted); font-size:12px; margin-top:24px; }}
  a {{ color:#1a73e8; }}
</style>
</head>
<body>
<main>
  <h1>Socio-economic drivers of health, England &amp; Wales</h1>
  <p class="sub">2011 and 2021 Census, {areas} local authorities ({matched} matched across both censuses).
     <a href="https://github.com/ishaan-bhalla/Visual_Analytics">Code and full write-up on GitHub</a></p>
  <div class="findings">
    <div class="card"><b>{improved} of {matched}</b>areas reported better health in 2021 than 2011 (mean +{mean:.2f} pp)</div>
    <div class="card"><b>Catch-up</b>areas that were less healthy in 2011 improved more (r = {corr:.2f})</div>
    <div class="card"><b>{r2:.0f}%</b>of the variation in change explained on held-out areas; age structure, not in this data, is a likely missing factor</div>
    <div class="card"><b>{diverse} areas</b>form a distinct diverse, high-migration cluster: London and large cities</div>
  </div>
  <p class="how">How to use: drag a box on the PCA plot, or click any area (shift-click for several), and the same
     areas light up in every view. Double-click to clear. The region menu filters everything. Grey areas on the
     change map are councils merged between 2019 and 2021, which have no like-for-like 2021 figure.</p>
  <div id="controls"></div>
  <div id="vis"></div>
  <footer>Data: ONS 2011 and 2021 Census via Nomis; boundaries: ONS Open Geography Portal, Local Authority
    Districts (December 2011). Contains public sector information licensed under the Open Government Licence v3.0
    and OS data &copy; Crown copyright and database right. Built with Python and Altair.</footer>
</main>
<script>
  vegaEmbed("#vis", {spec}, {{ actions: false }}).then(r => {{ window.vegaView = r.view; }}).catch(console.error);
</script>
</body>
</html>
"""


def main() -> None:
    DOCS.mkdir(exist_ok=True)
    FIGS.mkdir(exist_ok=True)
    df, summary, geo = load()
    dash, figures = build(df, summary, geo)
    gc = summary["good_change"]
    diverse = next(c["areas"] for c in summary["clusters"] if c["name"].startswith("Diverse"))
    html = PAGE.format(vega=alt.VEGA_VERSION, vegalite=alt.VEGALITE_VERSION, embed=alt.VEGAEMBED_VERSION,
                       spec=dash.to_json(indent=None), areas=summary["areas_2011"],
                       matched=summary["areas_matched_2021"], improved=gc["areas_improved"], mean=gc["mean_pp"],
                       corr=gc["corr_with_2011_health"], r2=summary["models"]["extended"]["cv_5fold"]["r2"] * 100,
                       diverse=diverse)
    (DOCS / "index.html").write_text(html)
    print(f"docs/index.html  {len(html) / 1e6:.1f} MB")
    dash.save(FIGS / "dashboard.png", scale_factor=2)
    split_rows(FIGS / "dashboard.png", ["maps", "explore", "model"])
    print("docs/figures/dashboard.png and one image per row")


def split_rows(png: Path, names: list[str]) -> None:
    """Cut the dashboard PNG into its rows at the widest horizontal white gaps."""
    from PIL import Image
    import numpy as np
    img = Image.open(png).convert("RGB")
    a = np.asarray(img)
    blank = (a > 250).all(axis=(1, 2))
    gaps, start = [], None
    for y, b in enumerate(blank):
        if b and start is None:
            start = y
        elif not b and start is not None:
            gaps.append((y - start, start, y))
            start = None
    cuts = sorted(sorted(gaps, reverse=True)[: len(names) - 1], key=lambda g: g[1])
    bounds = [0] + [(g[1] + g[2]) // 2 for g in cuts] + [a.shape[0]]
    for name, top, bottom in zip(names, bounds, bounds[1:]):
        img.crop((0, top, a.shape[1], bottom)).save(png.parent / f"{name}.png")


if __name__ == "__main__":
    main()
