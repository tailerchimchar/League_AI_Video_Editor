#!/usr/bin/env python3
"""Render top-down champion sprites from GLB 3D models using Blender.

Usage:
    blender --background --python scripts/render_topdown_sprites.py
    blender --background --python scripts/render_topdown_sprites.py -- --force
    blender --background --python scripts/render_topdown_sprites.py -- --batch A-F
    blender --background --python scripts/render_topdown_sprites.py -- --resolution 128

Reads GLB files from D:/LoLVideoAI/GLBs/{ChampionName}/{ChampionName}_skin{num}.glb
and renders top-down PNG sprites to data/champions/sprites/.

Output structure:
    data/champions/sprites/{ChampionName}.png          (default skin = skin0)
    data/champions/sprites/{ChampionName}_skin{num}.png (other skins)

Arguments after '--' are passed to the script:
    --force           Re-render existing sprites
    --batch X or X-Y  Only render champions starting with these letters
    --resolution N    Output image size in pixels (default: 128)
"""

import string
import sys
from pathlib import Path

try:
    import bpy
except ImportError:
    print("ERROR: This script must be run inside Blender.")
    print("Usage: blender --background --python scripts/render_topdown_sprites.py")
    sys.exit(1)

# Parse arguments after '--'
argv = sys.argv
if "--" in argv:
    script_args = argv[argv.index("--") + 1:]
else:
    script_args = []

force = "--force" in script_args

resolution = 128
if "--resolution" in script_args:
    idx = script_args.index("--resolution")
    if idx + 1 < len(script_args):
        resolution = int(script_args[idx + 1])

batch_filter = None
if "--batch" in script_args:
    idx = script_args.index("--batch")
    if idx + 1 < len(script_args):
        batch_str = script_args[idx + 1].upper().strip()
        if len(batch_str) == 1 and batch_str in string.ascii_uppercase:
            batch_filter = {batch_str}
        elif len(batch_str) == 3 and batch_str[1] == "-":
            start, end = batch_str[0], batch_str[2]
            batch_filter = {chr(c) for c in range(ord(start), ord(end) + 1)}
        else:
            print(f"ERROR: Invalid batch range '{batch_str}'. Use a single letter (S) or range (A-F).")
            sys.exit(1)

# Paths
REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = Path("D:/LoLVideoAI/GLBs")
SPRITES_DIR = REPO_ROOT / "data" / "champions" / "sprites"
SPRITES_DIR.mkdir(parents=True, exist_ok=True)


def clear_scene():
    """Remove all objects, meshes, materials, etc. from the scene."""
    bpy.ops.wm.read_factory_settings(use_empty=True)


def import_glb(filepath: str):
    """Import a GLB file and return the imported objects."""
    bpy.ops.import_scene.gltf(filepath=filepath)
    return [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]


def center_and_normalize(objects):
    """Move imported objects to the origin and normalize scale to fit in a unit sphere."""
    if not objects:
        return

    from mathutils import Vector

    # Compute combined bounding box
    min_co = [float("inf")] * 3
    max_co = [float("-inf")] * 3

    for obj in objects:
        for corner in obj.bound_box:
            world_co = obj.matrix_world @ Vector(corner)
            for i in range(3):
                min_co[i] = min(min_co[i], world_co[i])
                max_co[i] = max(max_co[i], world_co[i])

    # Center offset
    center = [(min_co[i] + max_co[i]) / 2.0 for i in range(3)]
    for obj in objects:
        obj.location.x -= center[0]
        obj.location.y -= center[1]
        obj.location.z -= center[2]

    # Normalize scale so the model fits in a 2x2x2 cube
    dims = [max_co[i] - min_co[i] for i in range(3)]
    max_dim = max(dims) if max(dims) > 0 else 1.0
    scale_factor = 2.0 / max_dim
    for obj in objects:
        obj.scale *= scale_factor


def setup_camera():
    """Set up an orthographic camera looking straight down (-Z)."""
    cam_data = bpy.data.cameras.new("TopDownCam")
    cam_data.type = "ORTHO"
    cam_data.ortho_scale = 2.5  # Slightly larger than the 2x2 normalized model

    cam_obj = bpy.data.objects.new("TopDownCam", cam_data)
    bpy.context.scene.collection.objects.link(cam_obj)

    # Position above, looking down
    cam_obj.location = (0, 0, 5)
    cam_obj.rotation_euler = (0, 0, 0)  # Default looks down -Z for ortho

    bpy.context.scene.camera = cam_obj
    return cam_obj


def setup_lighting():
    """Set up neutral lighting: ambient world + one soft directional light."""
    # World ambient light
    world = bpy.data.worlds.new("SpriteWorld")
    bpy.context.scene.world = world
    world.use_nodes = True
    bg_node = world.node_tree.nodes.get("Background")
    if bg_node:
        bg_node.inputs["Color"].default_value = (0.3, 0.3, 0.3, 1.0)
        bg_node.inputs["Strength"].default_value = 0.5

    # Soft directional light from above
    light_data = bpy.data.lights.new("TopLight", "SUN")
    light_data.energy = 2.0
    light_obj = bpy.data.objects.new("TopLight", light_data)
    bpy.context.scene.collection.objects.link(light_obj)
    light_obj.location = (0, 0, 10)
    light_obj.rotation_euler = (0, 0, 0)  # Pointing down


def setup_render(res: int):
    """Configure render settings for transparent background sprite output."""
    scene = bpy.context.scene
    # BLENDER_EEVEE_NEXT was only in Blender 4.x, removed in 5.0
    if "BLENDER_EEVEE_NEXT" in [e.identifier for e in bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items]:
        scene.render.engine = "BLENDER_EEVEE_NEXT"
    else:
        scene.render.engine = "BLENDER_EEVEE"
    scene.render.resolution_x = res
    scene.render.resolution_y = res
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = True
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"


def render_model(glb_path: Path, output_path: Path, res: int):
    """Render a single GLB model to a top-down sprite PNG."""
    clear_scene()

    # Import the model
    objects = import_glb(str(glb_path))
    if not objects:
        print(f"  WARNING: No mesh objects found in {glb_path.name}")
        return False

    center_and_normalize(objects)
    setup_camera()
    setup_lighting()
    setup_render(res)

    # Render
    bpy.context.scene.render.filepath = str(output_path)
    bpy.ops.render.render(write_still=True)
    return True


def main():
    if not MODELS_DIR.exists():
        print(f"ERROR: Models directory not found: {MODELS_DIR}")
        print("Run scripts/download_champion_models.py first.")
        sys.exit(1)

    # Discover champion folders with GLB files inside
    champ_dirs = sorted([
        d for d in MODELS_DIR.iterdir()
        if d.is_dir() and any(d.glob("*.glb"))
    ])

    if not champ_dirs:
        print(f"ERROR: No champion folders with GLB files found in {MODELS_DIR}")
        sys.exit(1)

    # Apply batch filter
    if batch_filter:
        champ_dirs = [d for d in champ_dirs if d.name[0].upper() in batch_filter]
        print(f"Batch filter: {', '.join(sorted(batch_filter))}")

    # Count total GLBs
    total_glbs = sum(len(list(d.glob("*.glb"))) for d in champ_dirs)
    print(f"Found {len(champ_dirs)} champions, {total_glbs} skin GLBs")
    print(f"Output: {SPRITES_DIR} ({resolution}x{resolution} px)")
    print()

    rendered = 0
    skipped = 0
    failed = 0

    for champ_dir in champ_dirs:
        champ_name = champ_dir.name
        glb_files = sorted(champ_dir.glob("*.glb"))

        champ_rendered = 0
        champ_skipped = 0
        champ_failed = 0

        for glb_path in glb_files:
            # Parse skin number from filename: {ChampName}_skin{num}.glb
            stem = glb_path.stem  # e.g. "Ahri_skin0"
            if "_skin" in stem:
                skin_num_str = stem.split("_skin")[-1]
                try:
                    skin_num = int(skin_num_str)
                except ValueError:
                    skin_num = 0
            else:
                skin_num = 0

            # Default skin (skin0) gets the clean name, others get _skin{num}
            if skin_num == 0:
                output_path = SPRITES_DIR / f"{champ_name}.png"
            else:
                output_path = SPRITES_DIR / f"{champ_name}_skin{skin_num}.png"

            if output_path.exists() and not force:
                champ_skipped += 1
                continue

            try:
                success = render_model(glb_path, output_path, resolution)
                if success:
                    champ_rendered += 1
                else:
                    champ_failed += 1
            except Exception as e:
                print(f"    ERROR: {glb_path.name}: {e}")
                champ_failed += 1

        # Per-champion summary
        status_parts = []
        if champ_rendered > 0:
            status_parts.append(f"{champ_rendered} rendered")
        if champ_skipped > 0:
            status_parts.append(f"{champ_skipped} existing")
        if champ_failed > 0:
            status_parts.append(f"{champ_failed} failed")
        status = ", ".join(status_parts) if status_parts else "nothing"
        print(f"  {champ_name} ({len(glb_files)} skins): {status}")

        rendered += champ_rendered
        skipped += champ_skipped
        failed += champ_failed

    print(f"\nDone!")
    print(f"  Sprites dir: {SPRITES_DIR}")
    print(f"  Total: {rendered} rendered, {skipped} existing, {failed} failed")


if __name__ == "__main__":
    main()
