def apply_chart_styling(fig, title=None):
    """Apply consistent styling to all charts"""
    layout_args = {
        "plot_bgcolor": "white",
        "paper_bgcolor": "white",
        "font": dict(family="Arial, sans-serif", size=12),
        "margin": dict(l=40, r=20, t=40, b=20)
    }
    if title:
        layout_args["title"] = title
    fig.update_layout(**layout_args)
    return fig