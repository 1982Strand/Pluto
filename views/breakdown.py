import streamlit as st
import plotly.graph_objects as go
from utils.formatting import _da_num, _flatten_html, format_quantity

def render_breakdown(rows, color_map, *, count_word="positioner", chart_height=320):
    """Render donut + farvet HTML-tabel for en bucket-fordeling.

    rows: liste af dicts med: name, count, value_dkk, invested_dkk, gain_dkk,
          gain_pct, alloc_pct, optional emoji
    color_map: dict[name -> rgb-streng]
    """
    col_chart, col_list = st.columns([1, 2])
    with col_chart:
        fig = go.Figure(go.Pie(
            labels=[r["name"] for r in rows],
            values=[r["value_dkk"] for r in rows],
            hole=0.6,
            marker=dict(
                colors=[color_map[r["name"]] for r in rows],
                line=dict(color="white", width=2),
            ),
            textinfo="none",
            hovertemplate="<b>%{label}</b><br>%{value:,.2f} DKK (%{percent})<extra></extra>",
        ))
        fig.update_layout(
            height=chart_height, showlegend=False,
            margin=dict(l=0, r=0, t=0, b=0),
            paper_bgcolor="white", plot_bgcolor="white",
            separators=".,",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_list:
        rows_html = ""
        for r in rows:
            c = color_map[r["name"]]
            gain_color = "#2e7d32" if r["gain_dkk"] >= 0 else "#d32f2f"
            arrow = "▲" if r["gain_dkk"] >= 0 else "▼"
            if abs(r["gain_dkk"]) < 0.01 and r.get("hide_zero_gain"):
                gain_html = "<small style='color:#888;'>—</small>"
            else:
                gain_html = (
                    f"{_da_num(r['gain_dkk'], signed=True)}<br>"
                    f"<small>{arrow} {_da_num(r['gain_pct'], signed=True)}%</small>"
                )
            emoji = (r.get("emoji", "") + " ") if r.get("emoji") else ""
            rows_html += f"""
            <tr>
              <td style='border-left:4px solid {c}; padding:10px 0 10px 14px;'>
                <strong>{emoji}{r['name']}</strong><br>
                <small style='color:#888;'>{r['count']} {count_word}</small>
              </td>
              <td style='padding:10px;'>
                <strong>{_da_num(r['value_dkk'])}</strong><br>
                <small style='color:#888;'>{_da_num(r['invested_dkk'])}</small>
              </td>
              <td style='padding:10px; color:{gain_color};'>
                {gain_html}
              </td>
              <td style='padding:10px; text-align:right;'>
                <strong>{_da_num(r['alloc_pct'])}%</strong>
              </td>
            </tr>
            """
        table_html = f"""
        <table style='width:100%; border-collapse:collapse; font-size:14px;'>
          <thead>
            <tr style='border-bottom:1px solid #ddd; color:#666;'>
              <th style='text-align:left; padding:8px 0 8px 14px;'>Navn</th>
              <th style='text-align:left; padding:8px;'>Værdi/Investeret</th>
              <th style='text-align:left; padding:8px;'>Gevinst</th>
              <th style='text-align:right; padding:8px;'>Allokering</th>
            </tr>
          </thead>
          <tbody>{rows_html}</tbody>
        </table>
        """
        st.markdown(_flatten_html(table_html), unsafe_allow_html=True)


def render_drilldown(rows, buckets_dict, color_map, *, count_word="positioner",
                     bucket_value_total_key="value_dkk"):
    """Render expander pr. bucket med tabel over underliggende positioner."""
    st.write("")
    st.caption(f"Klik for at se de underliggende {count_word}")
    for r in rows:
        bucket = buckets_dict.get(r["name"])
        if not bucket or not bucket.get("positions"):
            continue
        positions = sorted(bucket["positions"], key=lambda p: p["value_dkk"], reverse=True)
        bucket_total = bucket.get(bucket_value_total_key, 0) or 0
        color = color_map.get(r["name"], "rgb(150,150,150)")
        emoji = (r.get("emoji", "") + " ") if r.get("emoji") else ""
        label = (
            f"{emoji}{r['name']} — {r['count']} {count_word}  •  "
            f"{_da_num(r['value_dkk'])} DKK  •  "
            f"{_da_num(r['alloc_pct'])}%"
        )
        with st.expander(label):
            pos_rows_html = ""
            for p in positions:
                gc = "#2e7d32" if p["gain_dkk"] >= 0 else "#d32f2f"
                share = (p["value_dkk"] / bucket_total * 100) if bucket_total else 0
                qty_str = format_quantity(p.get("qty"))
                if abs(p["gain_dkk"]) < 0.01 and p.get("hide_zero_gain"):
                    gain_cell = "<small style='color:#888;'>—</small>"
                else:
                    gain_cell = (
                        f"{_da_num(p['gain_dkk'], signed=True)}<br>"
                        f"<small>{_da_num(p.get('pos_pct', 0), signed=True)}%</small>"
                    )
                pos_rows_html += f"""
                <tr>
                  <td style='border-left:4px solid {color}; padding:8px 0 8px 14px;'>
                    <strong>{p['name'][:40]}</strong><br>
                    <small style='color:#888;'>{p['ticker']} • {qty_str} stk.</small>
                  </td>
                  <td style='padding:8px;'>
                    <strong>{_da_num(p['value_dkk'])}</strong><br>
                    <small style='color:#888;'>{_da_num(p['invested_dkk'])}</small>
                  </td>
                  <td style='padding:8px; color:{gc};'>
                    {gain_cell}
                  </td>
                  <td style='padding:8px; text-align:right;'>
                    <strong>{_da_num(share)}%</strong>
                  </td>
                </tr>
                """
            pos_table_html = f"""
            <table style='width:100%; border-collapse:collapse; font-size:13px;'>
              <thead>
                <tr style='border-bottom:1px solid #ddd; color:#666;'>
                  <th style='text-align:left; padding:6px 0 6px 14px;'>Position</th>
                  <th style='text-align:left; padding:6px;'>Værdi/Investeret</th>
                  <th style='text-align:left; padding:6px;'>Gevinst</th>
                  <th style='text-align:right; padding:6px;'>Andel af gruppe</th>
                </tr>
              </thead>
              <tbody>{pos_rows_html}</tbody>
            </table>
            """
            st.markdown(_flatten_html(pos_table_html), unsafe_allow_html=True)
            
def _lighten_rgb(rgb_str, factor=0.45):
    """Bland 'rgb(r, g, b)' med hvid for at lave en lysere variant til sunburst-yderring."""
    if not isinstance(rgb_str, str) or not rgb_str.startswith("rgb"):
        return rgb_str
    try:
        nums = rgb_str.replace("rgb(", "").replace(")", "").split(",")
        r, g, b = (int(n.strip()) for n in nums[:3])
    except Exception:
        return rgb_str
    return (
        f"rgb({int(r + (255 - r) * factor)},"
        f"{int(g + (255 - g) * factor)},"
        f"{int(b + (255 - b) * factor)})"
    )