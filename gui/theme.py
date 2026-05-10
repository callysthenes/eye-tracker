"""
Cyberpunk-style PyQt5 theme with neon colors and dark aesthetic.
"""

import os

# Cyberpunk color palette
COLORS = {
    'background': '#0a0e27',      # Very dark blue
    'surface': '#1a1f3a',         # Dark blue-gray
    'surface_variant': '#2a2f4a', # Slightly lighter
    
    'primary_neon': '#00f0ff',    # Cyan/electric blue
    'secondary_neon': '#ff006e',  # Hot pink/magenta
    'tertiary_neon': '#d946ef',   # Purple
    'accent_neon': '#ffd700',     # Gold
    
    'success': '#00ff41',         # Neon green
    'warning': '#ff6b35',         # Orange-red
    'critical': '#ff003f',        # Deep red/magenta
    
    'text_primary': '#e0e0e0',    # Light gray
    'text_secondary': '#a0a0a0',  # Medium gray
    'border': '#00f0ff',          # Cyan borders
}


def get_stylesheet():
    """
    Get the complete cyberpunk QSS stylesheet.
    
    Returns:
        QSS stylesheet string
    """
    return f"""
    /* Main Window */
    QMainWindow {{
        background-color: {COLORS['background']};
        border: 2px solid {COLORS['primary_neon']};
    }}
    
    QWidget {{
        background-color: {COLORS['background']};
        color: {COLORS['text_primary']};
    }}
    
    /* Labels */
    QLabel {{
        color: {COLORS['text_primary']};
        font-family: 'Courier New', monospace;
        font-size: 11px;
    }}
    
    QLabel#title {{
        color: {COLORS['primary_neon']};
        font-size: 18px;
        font-weight: bold;
    }}
    
    QLabel#stat_label {{
        color: {COLORS['secondary_neon']};
        font-size: 12px;
        font-weight: bold;
    }}
    
    QLabel#warning {{
        color: {COLORS['warning']};
        font-weight: bold;
    }}
    
    QLabel#critical {{
        color: {COLORS['critical']};
        font-weight: bold;
    }}
    
    /* Buttons */
    QPushButton {{
        background-color: {COLORS['surface_variant']};
        color: {COLORS['primary_neon']};
        border: 2px solid {COLORS['primary_neon']};
        border-radius: 4px;
        padding: 6px 12px;
        font-family: 'Courier New', monospace;
        font-size: 10px;
        font-weight: bold;
        outline: none;
    }}
    
    QPushButton:hover {{
        background-color: {COLORS['surface']};
        border: 2px solid {COLORS['secondary_neon']};
        color: {COLORS['secondary_neon']};
    }}
    
    QPushButton:pressed {{
        background-color: {COLORS['primary_neon']};
        color: {COLORS['background']};
        border: 2px solid {COLORS['primary_neon']};
    }}
    
    QPushButton#alert_btn {{
        border: 2px solid {COLORS['critical']};
        color: {COLORS['critical']};
    }}
    
    QPushButton#alert_btn:hover {{
        border: 2px solid {COLORS['critical']};
        color: {COLORS['background']};
        background-color: {COLORS['critical']};
    }}
    
    /* Checkboxes and Radio Buttons */
    QCheckBox {{
        color: {COLORS['text_primary']};
        font-family: 'Courier New', monospace;
        font-size: 10px;
        spacing: 8px;
    }}
    
    QCheckBox::indicator {{
        width: 16px;
        height: 16px;
        border: 2px solid {COLORS['primary_neon']};
        background-color: {COLORS['surface']};
        border-radius: 2px;
    }}
    
    QCheckBox::indicator:checked {{
        background-color: {COLORS['primary_neon']};
    }}
    
    /* Frames and GroupBoxes */
    QFrame {{
        background-color: {COLORS['background']};
        border: 1px solid {COLORS['border']};
        border-radius: 4px;
    }}
    
    QGroupBox {{
        color: {COLORS['primary_neon']};
        border: 2px solid {COLORS['primary_neon']};
        border-radius: 4px;
        margin-top: 8px;
        padding-top: 8px;
        font-family: 'Courier New', monospace;
        font-size: 11px;
        font-weight: bold;
    }}
    
    QGroupBox::title {{
        subcontrol-origin: margin;
        left: 10px;
        padding: 0 3px 0 3px;
    }}
    
    /* Progress Bar */
    QProgressBar {{
        border: 2px solid {COLORS['primary_neon']};
        border-radius: 4px;
        text-align: center;
        background-color: {COLORS['surface']};
        color: {COLORS['primary_neon']};
        font-family: 'Courier New', monospace;
        font-size: 9px;
        font-weight: bold;
    }}
    
    QProgressBar::chunk {{
        background-color: {COLORS['primary_neon']};
        border-radius: 2px;
    }}
    
    /* Status Bar */
    QStatusBar {{
        background-color: {COLORS['surface']};
        color: {COLORS['text_secondary']};
        border-top: 1px solid {COLORS['primary_neon']};
        font-family: 'Courier New', monospace;
        font-size: 9px;
    }}
    """


def get_alert_stylesheet(level: str) -> str:
    """
    Get stylesheet for alert level indicator.
    
    Args:
        level: Alert level ('normal', 'warning', 'critical')
    
    Returns:
        QSS for the alert indicator
    """
    if level == 'critical':
        color = COLORS['critical']
        border_width = 3
    elif level == 'warning':
        color = COLORS['warning']
        border_width = 2
    else:
        color = COLORS['success']
        border_width = 2
    
    return f"""
    QFrame {{
        background-color: {COLORS['surface_variant']};
        border: {border_width}px solid {color};
        border-radius: 4px;
    }}
    """


if __name__ == "__main__":
    # Test: print color palette
    print("Cyberpunk Color Palette:")
    for name, color in COLORS.items():
        print(f"  {name}: {color}")
