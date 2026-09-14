"""Debug ray_cast in Blender to understand why visibility check fails."""
import bpy
import mathutils
import json
import math
import sys

# Parse args
argv = sys.argv
if "--" in argv:
    argv = argv[argv.index("--") + 1:]
    scene_id = argv[0] if argv else "3rscan__095821fb-e2c2-2de1-94df-20f2cb423bcb"
else:
    scene_id = "3rscan__095821fb-e2c2-2de1-94df-20f2cb423bcb"

composed_glb = f"/datadisk/NavBench3D/composed/{scene_id}.glb"
layout_path = f"/datadisk/NavBench3D/scenes/{scene_id}/layout.json"

# Clear scene
bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete(use_global=True)

# Load composed GLB
print(f"Loading: {composed_glb}")
bpy.ops.import_scene.gltf(filepath=composed_glb)
print(f"Loaded {len(bpy.data.objects)} objects")

# List all objects
for obj in bpy.data.objects:
    print(f"  Object: {obj.name}, type={obj.type}, loc={tuple(round(v,2) for v in obj.location)}")

# Set up camera  
cam_data = bpy.data.cameras.new("DebugCam")
cam_data.lens_unit = "FOV"
cam_data.angle = math.radians(65)
cam_obj = bpy.data.objects.new("DebugCam", cam_data)
bpy.context.scene.collection.objects.link(cam_obj)
bpy.context.scene.camera = cam_obj
cam_obj.location = (1.99, 1.07, 1.60)
pitch = math.pi / 2 - math.radians(8)
cam_obj.rotation_euler = (pitch, 0, 0)

# Update depsgraph
bpy.context.view_layer.update()
depsgraph = bpy.context.evaluated_depsgraph_get()

# Load layout
with open(layout_path) as f:
    layout = json.load(f)

cam_pos = mathutils.Vector((1.99, 1.07, 1.60))

print("\n=== Ray-cast debug ===")
for obj_info in layout:
    bbox = obj_info.get("bbox", [])
    if len(bbox) < 6 or not obj_info.get("model_uid", ""):
        continue
    
    ox, oy, oz = bbox[0], bbox[1], bbox[2]
    sx, sy, sz = bbox[3], bbox[4], bbox[5]
    category = obj_info.get("category", "?")
    
    target = mathutils.Vector((ox, oy, oz))
    direction = target - cam_pos
    dist = direction.length
    direction_norm = direction.normalized()
    
    # Ray cast
    hit, loc, norm, idx, hit_obj, mat = bpy.context.scene.ray_cast(
        depsgraph, cam_pos, direction_norm, distance=dist + 0.5
    )
    
    obj_max_r = math.sqrt(sx**2 + sy**2 + sz**2) / 2
    
    if hit:
        hit_dist = (loc - cam_pos).length
        diff = abs(hit_dist - dist)
        match = diff < obj_max_r
        hit_name = hit_obj.name if hit_obj else "?"
        print(f"  {category} (id={obj_info.get('id',0)}): "
              f"target_dist={dist:.2f}, hit_dist={hit_dist:.2f}, "
              f"diff={diff:.2f}, obj_r={obj_max_r:.2f}, "
              f"match={match}, hit_obj={hit_name}")
    else:
        print(f"  {category} (id={obj_info.get('id',0)}): NO HIT "
              f"(target_dist={dist:.2f})")

# Also try a simple downward ray to check floor
print("\n=== Floor check ===")
down = mathutils.Vector((0, 0, -1))
hit, loc, norm, idx, hit_obj, mat = bpy.context.scene.ray_cast(
    depsgraph, cam_pos, down, distance=5.0
)
if hit:
    print(f"Floor hit at z={loc.z:.3f}, object={hit_obj.name if hit_obj else '?'}")
else:
    print("No floor hit")
