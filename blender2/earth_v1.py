import bpy

# 地球の直径 (km)
earth_diameter = 12742

# Blender単位を1kmに設定（必要に応じてスケールを変更）
scale_factor = 1  # ここでは1 Blender単位 = 1 km としています

# 地球の直径に基づいて円の半径を計算
radius = (earth_diameter / 2) * scale_factor

# 新しい円を作成
bpy.ops.mesh.primitive_circle_add(radius=radius, location=(0, 0, 0))

# 作成した円のオブジェクトを取得
circle = bpy.context.object

# オブジェクト名を変更
circle.name = "EarthSizedCircle"

# 3Dビューのスペースデータを取得
for area in bpy.context.screen.areas:
    if area.type == 'VIEW_3D':
        for space in area.spaces:
            if space.type == 'VIEW_3D':
                # クリッピングの最小距離と最大距離を設定
                space.clip_start = 0.1  # 最小距離（必要に応じて変更）
                space.clip_end = 50000  # 最大距離（ここでは50,000 Blender単位に設定）

# import bpy

# # シーンをリセット
# bpy.ops.object.select_all(action='SELECT')
# bpy.ops.object.delete(use_global=False)

# # UV Sphereを作成
# bpy.ops.mesh.primitive_uv_sphere_add(radius=1, location=(0, 0, 0))
# sphere = bpy.context.active_object
# sphere.name = "Earth"

# # マテリアルを作成
# material = bpy.data.materials.new(name="EarthMaterial")
# material.use_nodes = True
# sphere.data.materials.append(material)

# # ノードツリーにアクセス
# nodes = material.node_tree.nodes
# links = material.node_tree.links

# # 既存のノードをクリア
# for node in nodes:
#     nodes.remove(node)

# # 必要なノードを追加
# output_node = nodes.new(type='ShaderNodeOutputMaterial')
# output_node.location = (400, 0)

# principled_node = nodes.new(type='ShaderNodeBsdfPrincipled')
# principled_node.location = (200, 0)

# # ノードを接続
# links.new(principled_node.outputs['BSDF'], output_node.inputs['Surface'])

# # マテリアルプロパティを設定
# principled_node.inputs['Base Color'].default_value = (0.5, 0.5, 0.5, 1)  # 灰色
# principled_node.inputs['Roughness'].default_value = 0.5  # 適度な粗さ
# principled_node.inputs['Specular'].default_value = 0.5  # 適度なスペキュラ反射

# # ライトを作成
# bpy.ops.object.light_add(type='SUN', location=(10, 10, 10))
# sun = bpy.context.active_object
# sun.data.energy = 10  # ライトの強さを調整

# # カメラを作成
# bpy.ops.object.camera_add(location=(0, -5, 0))
# camera = bpy.context.active_object
# camera.rotation_euler = (1.5708, 0, 0)  # カメラを地球に向ける

# # シーンのカメラを設定
# bpy.context.scene.camera = camera

