def apply_chart_styling(fig, title=None):
    """Wendet einheitliches Styling auf alle Diagramme an
    
    Args:
        fig: Das zu formatierende Plotly-Diagramm
        title: Optionaler Titel für das Diagramm
        
    Returns:
        Das formatierte Diagramm
    """
    # Standardeinstellungen für alle Diagramme
    layout_args = {
        "plot_bgcolor": "white",                 # Weißer Hintergrund für den Plotbereich
        "paper_bgcolor": "white",                # Weißer Hintergrund für das gesamte Diagramm
        "font": dict(family="Arial, sans-serif", size=12),  # Einheitliche Schriftart und -größe
        "margin": dict(l=40, r=20, t=40, b=20)  # Angepasste Ränder für optimale Platznutzung
    }
    
    # Fügt Titel hinzu, falls angegeben
    if title:
        layout_args["title"] = title
        
    # Wendet alle Layouteinstellungen auf das Diagramm an
    fig.update_layout(**layout_args)
    
    # Gibt das formatierte Diagramm zurück
    return fig
