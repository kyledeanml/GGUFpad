#!/bin/bash
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="$HOME/.local/share/applications"
ICONS_DIR="$HOME/.local/share/icons/hicolor/scalable/apps"
BIN_DIR="$HOME/.local/bin"

echo "=========================================================="
echo " Installing GGUFpanel for Linux Mint Cinnamon"
echo " (Local GGUF LLM Studio & Workstation)"
echo "=========================================================="

chmod +x "$DIR/run.sh" "$DIR/ggufpanel.py"

mkdir -p "$APP_DIR" "$ICONS_DIR" "$BIN_DIR" "$HOME/.icons" "$HOME/.local/share/pixmaps"

# Clean up previous ggufpad artifacts if present
rm -f "$APP_DIR/ggufpad.desktop" "$BIN_DIR/ggufpad" "$ICONS_DIR/ggufpad.svg" \
      "$HOME/.icons/ggufpad.svg" "$HOME/.local/share/pixmaps/ggufpad.svg" 2>/dev/null || true
rm -f "$HOME/.local/share/icons/hicolor/"*/apps/ggufpad.png 2>/dev/null || true

# 1. Install high-DPI vector icon & multi-size PNG icons
cp -f "$DIR/ggufpanel.svg" "$ICONS_DIR/ggufpanel.svg"
cp -f "$DIR/ggufpanel.svg" "$HOME/.icons/ggufpanel.svg"
cp -f "$DIR/ggufpanel.svg" "$HOME/.local/share/pixmaps/ggufpanel.svg"

python3 -B -c "
import os, gi
gi.require_version('GdkPixbuf', '2.0')
from gi.repository import GdkPixbuf
src = '$DIR/ggufpanel.svg'
base = os.path.expanduser('~/.local/share/icons/hicolor')
for sz in (16, 24, 32, 48, 64, 128, 256, 512):
    out_dir = os.path.join(base, f'{sz}x{sz}', 'apps')
    os.makedirs(out_dir, exist_ok=True)
    pb = GdkPixbuf.Pixbuf.new_from_file_at_scale(src, sz, sz, True)
    pb.savev(os.path.join(out_dir, 'ggufpanel.png'), 'png', [], [])
" 2>/dev/null || true

echo "[+] Installed GGUFpanel SVG & panel icon set: $ICONS_DIR/ggufpanel.svg"

# 2. Install executable launcher in ~/.local/bin
ln -sf "$DIR/run.sh" "$BIN_DIR/ggufpanel"
echo "[+] Created CLI executable link: $BIN_DIR/ggufpanel"

# 3. Install desktop launcher
cat <<EOF > "$APP_DIR/ggufpanel.desktop"
[Desktop Entry]
Name=GGUFpanel
Comment=GGUF model inspector, multi-format model browser, and local LLM studio for Linux Mint
GenericName=GGUF LLM Studio
Exec=$DIR/run.sh %F
Icon=$ICONS_DIR/ggufpanel.svg
Terminal=false
Type=Application
Categories=GTK;GNOME;Development;Utility;
MimeType=application/octet-stream;application/x-gguf;
StartupNotify=true
StartupWMClass=ggufpanel
Keywords=gguf;llm;ai;chat;llama;model;inspector;cuda;gpu;studio;
Actions=NewWindow;

[Desktop Action NewWindow]
Name=Open a New Window
Exec=$DIR/run.sh
EOF

chmod +x "$APP_DIR/ggufpanel.desktop"
echo "[+] Installed desktop entry: $APP_DIR/ggufpanel.desktop"

if command -v update-desktop-database >/dev/null 2>&1; then
    update-desktop-database "$APP_DIR" || true
fi

if command -v gtk-update-icon-cache >/dev/null 2>&1; then
    gtk-update-icon-cache -f -t "$HOME/.local/share/icons/hicolor" 2>/dev/null || true
fi

if command -v gsettings >/dev/null 2>&1; then
    python3 -B -c "
import gi
from gi.repository import Gio
try:
    settings = Gio.Settings.new('org.cinnamon')
    favs = [x for x in settings.get_strv('favorite-apps') if x != 'ggufpad.desktop']
    if 'ggufpanel.desktop' not in favs:
        favs.append('ggufpanel.desktop')
    settings.set_strv('favorite-apps', favs)
    print('[+] Added GGUFpanel to Cinnamon favorite apps')
except Exception:
    pass
" || true
fi

echo ""
echo "=========================================================="
echo " GGUFpanel Installation Complete!"
echo "=========================================================="
echo "You can launch GGUFpanel anytime from your Cinnamon Menu"
echo "or from terminal by typing:"
echo "  ggufpanel"
echo "=========================================================="
