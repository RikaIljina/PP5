import sys
from colorama import Fore, Style
from matplotlib.font_manager import FontProperties


def print_err(message):
    """Styling for error messages"""

    print(f"{Fore.RED}ERROR:{Style.RESET_ALL}\n{message}")


def print_warn(message):
    """Styling for warning messages"""

    print(f"{Fore.YELLOW}WARNING:{Style.RESET_ALL}\n{message}")
    

def get_user_confirmation():
    """User prompt helper function"""

    sys.stdout.flush()
    return input().lower()


def show_progress(list_len):
    """Show progress bar while loading images into memory"""
    width = 100
    for i in range(list_len):
        yield f'\r{''}: {"#"*int(width if i == list_len-1 else i//(list_len/width)):<{width}}|| '


def set_ticks(ax):
    """Remove ticks from figures"""

    ax.set_xticks([])
    ax.set_yticks([])


def font_props():
    """Set font properties for figures"""

    font_top = FontProperties(
    weight="bold",
    size=20,
    )
    title_props_box = {
    "y": 0.87,
    "va": "bottom",
    "bbox": {"facecolor": "lightgreen", "alpha": 0.3, "edgecolor": "green"},
    }
    title_props = {
    "y": 0.85,
    "va": "bottom",
    }
    return font_top, title_props_box, title_props


def style_dataframe(df, hl_label=None):
    """Custom dataframe style with the possibility to highlight a label"""

    col_names = [" "]
    col_names.extend(list(df.columns))
    for x, c in enumerate(col_names):
        if x == 0 and df.index.name is None:
            print(f"\033[1m{c:>16}\033[0m", end="")
        else:
            if x == 0:
                print(f"\033[1m{df.index.name:>16}\033[0m", end="")
            else:
                print(f"\033[1m{c.upper():>13}\033[0m", end="")
    print(f'\n{"-"*70}')
    for i in df.index:
        if i == hl_label:
            hl = Fore.GREEN
        else:
            hl = ""
        print(f"\033[1m{hl}{i:<16}\033[0m", end="")
        for c in df.columns:
            print(f"{hl if c == 'train' else ''}{df[c][i]:>13}\033[0m", end="")
        print("")