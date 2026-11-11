# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Replace tutorial assets with the saved training assets and verify geometry.
Only the test stage is edited. No training repository or upstream USD is modified.
"""
import hashlib
import json
import math
import re
from pathlib import Path
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, PhysxSchema
from scene_security import check_asset_endpoint

PATHS = {
    'robot': '/World/envs/env_0/Robot',
    'dp_plug': '/World/envs/env_0/DisplayPortPlug',
    'dp_socket': '/World/envs/env_0/DisplayPortSocket',
}



def geometry(root):
    records = {}
    for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
        rel = str(prim.GetPath().MakeRelativePath(root.GetPath()))
        attrs = {}
        for attr in prim.GetAttributes():
            n = attr.GetName()
            is_geometry = n in (
                'points', 'faceVertexCounts', 'faceVertexIndices', 'normals',
                'subdivisionScheme', 'visibility', 'purpose', 'orientation',
            )
            is_transform = prim != root and n.startswith('xformOp')
            if is_geometry or n.startswith('primvars:') or is_transform:
                attrs[n] = hashlib.sha256(str(attr.Get()).encode()).hexdigest()
        if attrs or prim.GetTypeName() == 'Mesh':
            records[rel] = {'type': prim.GetTypeName(), 'attrs': attrs}
    return records


def appearance(root):
    records = {}

    def normalize(value):
        if isinstance(value, Sdf.AssetPath):
            return value.resolvedPath or value.path
        return str(value).replace(str(root.GetPath()), '<asset>')
    for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        material = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()[0]
        content = {}
        if material:
            for item in Usd.PrimRange(material.GetPrim(), Usd.TraverseInstanceProxies()):
                rel = str(item.GetPath().MakeRelativePath(material.GetPath()))
                content[rel] = {
                    attr.GetName(): {
                        'value': normalize(attr.Get()),
                        'connections': [normalize(value) for value in attr.GetConnections()],
                    }
                    for attr in item.GetAttributes()
                    if attr.GetName().startswith(('inputs:', 'outputs:', 'info:'))
                }
        records[str(prim.GetPath().MakeRelativePath(root.GetPath()))] = content
    return records


def collision_definitions(root):
    records = {}
    for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        records[str(prim.GetPath().MakeRelativePath(root.GetPath()))] = {
            attr.GetName(): str(attr.Get())
            for attr in prim.GetAttributes()
            if attr.GetName() in ('physics:collisionEnabled', 'physics:approximation')
            or attr.GetName().startswith('physxSDFMeshCollision:')
        }
    return records



def differences(a, b):
    return [k for k in sorted(set(a) | set(b)) if a.get(k) != b.get(k)]


def set_attr(prim, name, value):
    attr = prim.GetAttribute(name)
    if not attr:
        if isinstance(value, bool):
            typ = Sdf.ValueTypeNames.Bool
        elif isinstance(value, int):
            typ = Sdf.ValueTypeNames.Int
        else:
            typ = Sdf.ValueTypeNames.Float
        attr = prim.CreateAttribute(name, typ)
    attr.Set(value)


def camel(name):
    words = name.split('_')
    return words[0] + ''.join((w.title() for w in words[1:]))


def apply_properties(prim, config, namespace, standard=()):
    for key, value in (config or {}).items():
        if key.startswith('_') or value is None:
            continue
        if key == 'mesh_collision_property':
            continue
        ns = 'physics' if key in standard else namespace
        set_attr(prim, ns + ':' + camel(key), value)


def rebuild(stage, manifest_path, cache_dir, report_path):
    """Rebuild and audit training assets in memory, retaining cached USDs and an audit."""
    manifest_path, cache_dir, report_path = map(Path, (manifest_path, cache_dir, report_path))
    report_path.parent.mkdir(parents=True, exist_ok=True)
    data = json.loads(manifest_path.read_text())
    if hashlib.sha256(Path(data['training_env_file']).read_bytes()).hexdigest() != data['training_env_sha256']:
        raise RuntimeError('Saved training configuration changed; regenerate the asset manifest')
    reference_dir = cache_dir.resolve()
    reference_dir.mkdir(parents=True, exist_ok=True)
    stage.SetEditTarget(stage.GetRootLayer())
    for extra in ('/World/DisplayPortStartFixture', '/World/DisplayPortSocketSupportBase'):
        stage.RemovePrim(extra)
    report = {
        'training_env_file': data['training_env_file'],
        'assets': {},
        'note': (
            'Geometry verified against assets resolved from saved training URLs now. '
            'Original training-time asset hashes were not recorded.'
        ),
    }
    for name, path in PATHS.items():
        cfg = data['assets'][name]
        spawn = cfg['spawn']
        url = check_asset_endpoint(spawn['usd_path'])
        print('TRAINING_ASSET_LOAD', name, url, flush=True)
        source = Usd.Stage.Open(url)
        if source is None or not source.GetDefaultPrim():
            raise RuntimeError('Unable to load training asset ' + url)
        expected = geometry(source.GetDefaultPrim())
        before = geometry(stage.GetPrimAtPath(path))
        original = stage.GetPrimAtPath(path)
        before_visual_differences = differences(appearance(source.GetDefaultPrim()), appearance(original))
        before_collision_differences = differences(collision_definitions(source.GetDefaultPrim()), collision_definitions(original))
        old_materials = []
        for collider in Usd.PrimRange(original, Usd.TraverseInstanceProxies()):
            if not collider.HasAPI(UsdPhysics.CollisionAPI):
                continue
            material = UsdShade.MaterialBindingAPI(collider).ComputeBoundMaterial(materialPurpose='physics')[0]
            values = {}
            for attr_name in ('physics:staticFriction', 'physics:dynamicFriction', 'physics:restitution'):
                attr = material.GetPrim().GetAttribute(attr_name) if material else None
                values[attr_name] = attr.Get() if attr else None
            old_materials.append({'collider': str(collider.GetPath()), 'material': str(material.GetPath()) if material else None, 'values': values})
        pinned = reference_dir / (name + '-' + hashlib.sha256(url.encode()).hexdigest()[:12] + '.usdc')
        if pinned.is_symlink():
            raise ValueError('Asset cache entry must not be a symbolic link')
        if not pinned.exists() and (not source.Flatten().Export(str(pinned))):
            raise RuntimeError('Failed to pin ' + name)
        stage.SetEditTarget(stage.GetRootLayer())
        stage.RemovePrim(path)
        prim = stage.DefinePrim(path, 'Xform')
        prim.GetReferences().AddReference(pinned.as_uri(), source.GetDefaultPrim().GetPath())
        # Root pose/scale must come from training, not stronger tutorial opinions.
        q = cfg['init_state']['rot']
        transform = Gf.Transform()
        transform.SetTranslation(Gf.Vec3d(*cfg['init_state']['pos']))
        transform.SetRotation(Gf.Rotation(Gf.Quatd(q[3], Gf.Vec3d(*q[:3]))))
        transform.SetScale(Gf.Vec3d(*(spawn['scale'] or [1, 1, 1])))
        UsdGeom.Xformable(prim).MakeMatrixXform().Set(transform.GetMatrix())
        # Uninstance to author saved rigid-body/collider overrides.
        for item in list(Usd.PrimRange(prim)):
            if item.IsInstance():
                item.SetInstanceable(False)
        for item in Usd.PrimRange(prim):
            if item.HasAPI(UsdPhysics.RigidBodyAPI):
                PhysxSchema.PhysxRigidBodyAPI.Apply(item)
                apply_properties(item, spawn['rigid_props'], 'physxRigidBody', ('rigid_body_enabled', 'kinematic_enabled'))
                if spawn.get('mass_props'):
                    UsdPhysics.MassAPI.Apply(item)
                    apply_properties(item, spawn['mass_props'], 'physics')
                if spawn.get('activate_contact_sensors'):
                    PhysxSchema.PhysxContactReportAPI.Apply(item).CreateThresholdAttr(0)
            if item.HasAPI(UsdPhysics.CollisionAPI):
                PhysxSchema.PhysxCollisionAPI.Apply(item)
                apply_properties(item, spawn['collision_props'], 'physxCollision', ('collision_enabled',))
            if item.HasAPI(UsdPhysics.ArticulationRootAPI) and spawn.get('articulation_props'):
                PhysxSchema.PhysxArticulationAPI.Apply(item)
                apply_properties(item, spawn['articulation_props'], 'physxArticulation', ('articulation_enabled',))
            if name == 'robot' and item.IsA(UsdPhysics.RevoluteJoint):
                joint = item.GetName()
                for actuator in cfg['actuators'].values():
                    if not any((re.fullmatch(p, joint) for p in actuator['joint_names_expr'])):
                        continue
                    drive = UsdPhysics.DriveAPI.Apply(item, 'angular')
                    drive.CreateStiffnessAttr(float(actuator['stiffness']) * math.pi / 180)
                    drive.CreateDampingAttr(float(actuator['damping']) * math.pi / 180)
                    force = actuator.get('joint_effort_limit')
                    if force is None:
                        force = actuator.get('effort_limit_sim')
                    speed = actuator.get('joint_velocity_limit')
                    if speed is None:
                        speed = actuator.get('velocity_limit_sim')
                    if force is not None:
                        drive.CreateMaxForceAttr(force)
                    PhysxSchema.PhysxJointAPI.Apply(item)
                    if speed is not None:
                        set_attr(item, 'physxJoint:maxJointVelocity', speed * 180 / math.pi)
                    if actuator['armature'] is not None:
                        set_attr(item, 'physxJoint:armature', actuator['armature'])
                    if actuator['friction'] is not None:
                        set_attr(item, 'physxJoint:jointFriction', actuator['friction'])
        matcfg = data['materials'][{'dp_plug': 'plug', 'dp_socket': 'socket', 'robot': 'robot'}[name]]
        materials = {}
        material_count = 0
        for item in Usd.PrimRange(prim):
            if not item.HasAPI(UsdPhysics.CollisionAPI):
                continue
            body = item
            while body and (not body.HasAPI(UsdPhysics.RigidBodyAPI)):
                body = body.GetParent()
            if not body or not re.fullmatch(matcfg['asset_cfg']['body_names'], body.GetName()):
                continue
            old = UsdShade.MaterialBindingAPI(item).ComputeBoundMaterial(materialPurpose='physics')[0]
            oldpath = str(old.GetPath()) if old else ''
            if oldpath not in materials:
                mat = UsdShade.Material.Define(stage, f'/World/TrainingPhysicsMaterials/{name}_{len(materials)}')
                # Preserve upstream combine modes; training changes only these
                # three material properties in its startup event.
                if old:
                    mat.GetPrim().GetReferences().AddInternalReference(old.GetPath())
                api = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
                api.CreateStaticFrictionAttr(matcfg['static_friction_range'][0])
                api.CreateDynamicFrictionAttr(matcfg['dynamic_friction_range'][0])
                api.CreateRestitutionAttr(matcfg['restitution_range'][0])
                materials[oldpath] = mat
            UsdShade.MaterialBindingAPI.Apply(item).Bind(materials[oldpath], materialPurpose='physics', bindingStrength=UsdShade.Tokens.strongerThanDescendants)
            material_count += 1
        actual = geometry(prim)
        mismatches = differences(expected, actual)
        visual_differences = differences(appearance(source.GetDefaultPrim()), appearance(prim))
        collision_differences = differences(collision_definitions(source.GetDefaultPrim()), collision_definitions(prim))
        old_detail = {}
        for key in differences(expected, before):
            training_attrs = expected.get(key, {}).get('attrs', {})
            tutorial_attrs = before.get(key, {}).get('attrs', {})
            old_detail[key] = {
                'missing_from_training': key not in expected,
                'missing_from_tutorial': key not in before,
                'changed_attributes': [
                    attr for attr in sorted(set(training_attrs) | set(tutorial_attrs))
                    if training_attrs.get(attr) != tutorial_attrs.get(attr)
                ],
            }
        property_checks = []
        for item in Usd.PrimRange(prim):
            groups = []
            if item.HasAPI(UsdPhysics.RigidBodyAPI):
                groups.append((spawn['rigid_props'], 'physxRigidBody', ('rigid_body_enabled', 'kinematic_enabled')))
                if spawn.get('mass_props'):
                    groups.append((spawn['mass_props'], 'physics', ()))
            if item.HasAPI(UsdPhysics.CollisionAPI):
                groups.append((spawn['collision_props'], 'physxCollision', ('collision_enabled',)))
            if item.HasAPI(UsdPhysics.ArticulationRootAPI) and spawn.get('articulation_props'):
                groups.append((spawn['articulation_props'], 'physxArticulation', ('articulation_enabled',)))
            for config, namespace, standard in groups:
                for key, value in (config or {}).items():
                    if key.startswith('_') or value is None or key == 'mesh_collision_property':
                        continue
                    attr = ('physics' if key in standard else namespace) + ':' + camel(key)
                    got = item.GetAttribute(attr).Get()
                    match = got == value if isinstance(value, bool) else math.isclose(float(got), float(value), rel_tol=1e-06, abs_tol=1e-09)
                    property_checks.append({'path': str(item.GetPath()), 'attribute': attr, 'expected': value, 'actual': got, 'match': match})
            if item.HasAPI(UsdPhysics.CollisionAPI):
                body = item
                while body and (not body.HasAPI(UsdPhysics.RigidBodyAPI)):
                    body = body.GetParent()
                if body and re.fullmatch(matcfg['asset_cfg']['body_names'], body.GetName()):
                    mat = UsdShade.MaterialBindingAPI(item).ComputeBoundMaterial(materialPurpose='physics')[0]
                    for field in ('static_friction', 'dynamic_friction', 'restitution'):
                        target = matcfg[field + '_range'][0]
                        got = mat.GetPrim().GetAttribute('physics:' + camel(field)).Get()
                        property_checks.append({'path': str(item.GetPath()), 'attribute': 'material:' + field, 'expected': target, 'actual': got, 'match': math.isclose(got, target, rel_tol=1e-06, abs_tol=1e-09)})
        property_failures = [r for r in property_checks if not r['match']]
        report['assets'][name] = {
            'old_visual_material_differing_meshes': before_visual_differences,
            'old_collision_definition_differing_paths': before_collision_differences,
            'old_bound_physics_materials': old_materials,
            'visual_material_differing_meshes': visual_differences,
            'collision_definition_differing_paths': collision_differences,
            'property_checks': len(property_checks),
            'property_failures': property_failures,
            'old_scene_difference_details': old_detail,
            'training_url': url,
            'pinned_asset': str(pinned),
            'sha256': hashlib.sha256(pinned.read_bytes()).hexdigest(),
            'training_geometry_records': len(expected),
            'old_scene_differing_paths': differences(expected, before),
            'corrected_scene_differing_paths': mismatches,
            'geometry_match': not mismatches,
            'physics_material_colliders': material_count,
            'spawn': spawn,
        }
        print('TRAINING_ASSET_VERIFIED', name, 'old differences', len(differences(expected, before)), 'new', len(mismatches), flush=True)
        if visual_differences or collision_differences or property_failures:
            report_path.write_text(json.dumps(report, indent=2))
            raise RuntimeError(f'{name} asset parity failed: visual={visual_differences}, collision={collision_differences}, properties={property_failures[:3]}')
        if mismatches:
            report_path.write_text(json.dumps(report, indent=2))
            raise RuntimeError(f'{name} geometry mismatch: {mismatches[:10]}')
    report_path.write_text(json.dumps(report, indent=2))
    return report
