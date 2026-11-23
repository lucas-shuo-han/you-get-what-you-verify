"""Tile blind images 6 per sheet, labelled only with blind id (sorted by id)."""
from pathlib import Path
from PIL import Image, ImageDraw
D = Path(__file__).parent; ims = sorted((D / "blind").glob("*.png")); (D / "blind_sheets").mkdir(exist_ok=True)
for s in range(0, len(ims), 6):
    chunk = ims[s:s+6]; tiles = [Image.open(p).convert("RGB") for p in chunk]
    w, h = tiles[0].size; sheet = Image.new("RGB", (w*2, (h+30)*3), "white"); d = ImageDraw.Draw(sheet)
    for i, (p, t) in enumerate(zip(chunk, tiles)):
        x, y = (i % 2)*w, (i//2)*(h+30); d.text((x+10, y+5), p.stem, fill="black"); sheet.paste(t.resize((w, h)), (x, y+30))
    sheet.save(D / "blind_sheets" / f"sheet_{s//6:02d}.png")
