"""Verify rendered images: check pixel statistics, non-black percentage, depth range."""
import os
import sys
import numpy as np
from PIL import Image

RENDER_DIR = "/datadisk/NavBench3D/renders/3rscan__095821fb-e2c2-2de1-94df-20f2cb423bcb"
OUTPUT_DIR = "/datadisk/NavBench3D/debug_compose"

def analyze_image(path, name):
    img = np.array(Image.open(path))
    non_black = np.mean(np.any(img > 10, axis=-1) if img.ndim == 3 else (img > 10)) * 100
    print(f"  {name}: shape={img.shape}, dtype={img.dtype}, "
          f"min={img.min()}, max={img.max()}, mean={img.mean():.1f}, "
          f"non-black={non_black:.1f}%")
    return img

def main():
    from PIL import Image as PILImage

    views = sorted([d for d in os.listdir(RENDER_DIR) if d.startswith("view_")])
    print(f"Found {len(views)} views")

    all_rgbs = []
    for view in views:
        view_dir = os.path.join(RENDER_DIR, view)
        print(f"\n{view}:")
        rgb = analyze_image(os.path.join(view_dir, "rgb.png"), "RGB")
        depth = analyze_image(os.path.join(view_dir, "depth.png"), "Depth")
        normal = analyze_image(os.path.join(view_dir, "normal.png"), "Normal")
        all_rgbs.append(rgb)

    # Check topdown
    topdown_path = os.path.join(RENDER_DIR, "topdown.png")
    if os.path.exists(topdown_path):
        print(f"\nTopdown:")
        analyze_image(topdown_path, "Topdown")

    # Create composite image: 2x2 grid of RGB views + topdown
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Resize all to same size for grid
    grid_size = 512
    grid = PILImage.new('RGB', (grid_size * 3, grid_size * 2), (30, 30, 30))

    for i, rgb in enumerate(all_rgbs):
        pil_img = PILImage.fromarray(rgb[:, :, :3] if rgb.ndim == 3 and rgb.shape[2] == 4 else rgb)
        pil_img = pil_img.resize((grid_size, grid_size))
        row, col = divmod(i, 3)
        grid.paste(pil_img, (col * grid_size, row * grid_size))

    if os.path.exists(topdown_path):
        td = PILImage.open(topdown_path).resize((grid_size, grid_size))
        grid.paste(td, (2 * grid_size, grid_size))

    # Add matplotlib topdown if exists
    mpl_td = os.path.join(OUTPUT_DIR, "topdown_zup_official.png")
    if os.path.exists(mpl_td):
        mtd = PILImage.open(mpl_td).resize((grid_size, grid_size))
        grid.paste(mtd, (1 * grid_size, grid_size))

    grid_path = os.path.join(OUTPUT_DIR, "render_verification_grid.png")
    grid.save(grid_path)
    print(f"\nSaved verification grid to {grid_path}")

if __name__ == "__main__":
    main()
