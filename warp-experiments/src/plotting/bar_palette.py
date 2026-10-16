from __future__ import annotations

BAR_PALETTE_CHOICES = ("paper", "blue", "red")

_BLUE_PALETTES = {
    3: ["#ece7f2", "#a6bddb", "#2b8cbe"],
    4: ["#f1eef6", "#bdc9e1", "#74a9cf", "#0570b0"],
    5: ["#f1eef6", "#bdc9e1", "#74a9cf", "#2b8cbe", "#045a8d"],
    6: ["#f1eef6", "#d0d1e6", "#a6bddb", "#74a9cf", "#2b8cbe", "#045a8d"],
    7: ["#f1eef6", "#d0d1e6", "#a6bddb", "#74a9cf", "#3690c0", "#0570b0", "#034e7b"],
    8: ["#fff7fb", "#ece7f2", "#d0d1e6", "#a6bddb", "#74a9cf", "#3690c0", "#0570b0", "#034e7b"],
    9: ["#fff7fb", "#ece7f2", "#d0d1e6", "#a6bddb", "#74a9cf", "#3690c0", "#0570b0", "#045a8d", "#023858"],
}

_RED_PALETTES = {
    3: ["#fdcc8a", "#fc8d59", "#d7301f"],
    4: ["#fef0d9", "#fdcc8a", "#fc8d59", "#d7301f"],
    5: ["#fef0d9", "#fdcc8a", "#fc8d59", "#e34a33", "#b30000"],
    6: ["#fef0d9", "#fdd49e", "#fdbb84", "#fc8d59", "#e34a33", "#b30000"],
    7: ["#fef0d9", "#fdd49e", "#fdbb84", "#fc8d59", "#ef6548", "#d7301f", "#990000"],
    8: ["#fff7ec", "#fee8c8", "#fdd49e", "#fdbb84", "#fc8d59", "#ef6548", "#d7301f", "#990000"],
    9: ["#fff7ec", "#fee8c8", "#fdd49e", "#fdbb84", "#fc8d59", "#ef6548", "#d7301f", "#b30000", "#7f0000"],
}

_FAMILIES = {
    "blue": _BLUE_PALETTES,
    "red": _RED_PALETTES,
}


def get_bar_palette(family: str, group_size: int) -> list[str]:
    family = family.lower()
    if family not in _FAMILIES:
        raise ValueError(f"unknown palette family: {family}")
    if group_size < 1:
        raise ValueError(f"group_size must be positive, got {group_size}")

    palettes = _FAMILIES[family]
    if group_size == 1:
        return [palettes[3][-1]]
    if group_size == 2:
        return palettes[3][-2:]
    if group_size not in palettes:
        raise ValueError(f"unsupported group_size={group_size} for family={family}")
    return list(palettes[group_size])


def palette_suffix(family: str) -> str:
    family = family.lower()
    if family == "paper":
        return ""
    return f"_{family}_palette"
