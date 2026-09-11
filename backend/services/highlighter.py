from docx.enum.text import WD_COLOR_INDEX


def get_highlight_color(probability):

    if probability >= 0.85:
        return WD_COLOR_INDEX.RED

    elif probability >= 0.60:
        return WD_COLOR_INDEX.YELLOW

    return None