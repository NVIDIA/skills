# Call examples

Every example is a complete MCP tool call. Effect ids and `params` keys mirror the app's UI;
the authoritative list for the running build is in the server `instructions`.

## Effects

Turn on mic noise removal:
```json
{ "name": "set_effects", "arguments": { "effects": [{ "effectId": "noise_removal", "enabled": true }] } }
```

Blur the camera background at 70% strength:
```json
{ "name": "set_effects", "arguments": { "effects": [{ "effectId": "background_blur", "enabled": true, "params": { "strength": 0.7 } }] } }
```

Warm virtual key light:
```json
{ "name": "set_effects", "arguments": { "effects": [{ "effectId": "virtual_key_light", "enabled": true, "params": { "warmth": 0.8 } }] } }
```

Increase camera frame rate with the balanced Frame Generation model:
```json
{ "name": "set_effects", "arguments": { "effects": [{ "effectId": "video_frame_generation", "enabled": true, "params": { "multiplier": 2, "model": "balanced" } }] } }
```

Enable Studio Voice with the full mic profile:
```json
{ "name": "set_effects", "arguments": { "effects": [{ "effectId": "studio_voice", "enabled": true, "params": { "micProfile": "full" } }] } }
```

Replace the camera background with a local image:
```json
{ "name": "set_effects", "arguments": { "effects": [{ "effectId": "background_replace", "enabled": true, "params": { "imagePath": "C:\\Users\\me\\Pictures\\background.png" } }] } }
```

"Clean up my mic and blur my background at 70%" — one call, two effects:
```json
{ "name": "set_effects", "arguments": { "effects": [
  { "effectId": "noise_removal", "enabled": true },
  { "effectId": "background_blur", "enabled": true, "params": { "strength": 0.7 } }
] } }
```

Resume the Live effects that Files processing suspended:
```json
{ "name": "set_effects", "arguments": { "action": "restore_previous_live" } }
```

## State

What's currently on?
```json
{ "name": "get_broadcast_state", "arguments": {} }
```

## Devices and resolution

List all cameras with resolutions:
```json
{ "name": "list_devices", "arguments": { "kind": "camera" } }
```

Switch to a specific camera at 1080p 60fps in one call:
```json
{ "name": "set_active_device", "arguments": { "kind": "camera", "deviceId": "<id from list_devices>", "width": 1920, "height": 1080, "frameRate": 60 } }
```

Change active camera resolution to 720p 30fps:
```json
{ "name": "set_camera_resolution", "arguments": { "deviceId": "<id>", "width": 1280, "height": 720, "frameRate": 30 } }
```

Switch microphone:
```json
{ "name": "set_active_device", "arguments": { "kind": "microphone", "deviceId": "<id from list_devices>" } }
```
