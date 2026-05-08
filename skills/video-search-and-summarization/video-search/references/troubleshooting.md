# Troubleshooting feedback loop

Isolate the problem encountered in video-search then iterate to resolve it. Examples of useful flows below.

## Gotchas

- ALWAYS use the method to list video sources with VST first with `vios`, before making curl requests to check Elasticsearch embeddings.
- If the video source is not ingested yet, NEVER use the VST upload APIs because they will not generate embeddings. Use the `videos-for-search` endpoint described below for video files (or `rtsp-streams/add` for RTSP streams), and use the term "ingest" instead of "upload" to avoid confusions
- NEVER try to guess the URL or VST API to check what is available in the system. Use the `vios` skill instead to list video sources and manage streams feeding into the search pipeline
```bash
# NEVER guess commands like
# curl -s "http://<ip>:30888/vst/api/v1/sensors" 
# curl -s "http://<ip>:30888/vst/api/v2/sensors?pageSize=50"
```

## Failure modes or unexpected results

- Video source(s) not returned or empty results
- Video source(s) returned, all with low similarity scores and/or a few with high scores. But sensor/stream names do not match the user query. Hence, not certain if these are correct answers, needs further verifications.
- Errors due to backend services all or partially not working

## Troubleshooting flows

Target specific components. Infer from the conversation where (`${HOST_IP}`, `${PORT}`) the service or model in question runs when running the commands below. If unable to infer, ask user to know `${HOST_IP}` and `${PORT}`.

The components in the externally accessible section should be reachable by their `${HOST_IP}`. But if they are not (ports blocked by firewall for security), ask user if they are accessible via ssh and run those commands through ssh. Otherwise ask user how they prefer to reach them.

If further investigation is required, refer to the full components from the `deploy` skill and choose which one to investigate.

### Externally accessible

- Ensure VST is running and ensure video source(s) of interest were ingested by listing them in VST via the `vios` skill.
  If not, offer the user the option to ingest them via the full pipeline API `videos-for-search` below if they are video files (or `rtsp-streams/add` for RTSP streams)

- If a video source in the system has no embeddings, it means it has not been ingested through the full pipeline. STOP and ask user if video can be re-ingested and if user can provide video source. If yes, carefully follow:
    - First delete it (avoid two copies) with indexes cleanup:
```bash
# For video files
# video_id = sensor / video UUID, same ID as in VST
curl -s -X DELETE "http://${HOST_IP}:8000/api/v1/videos/<video_id>" | jq .

# For RTSP streams
curl -s -X DELETE "http://${HOST_IP}:8000/api/v1/rtsp-streams/delete/<name>" | jq .
```
    - Then ingest video source again with the ingestion API:
```bash
# :8000 designate the VSS agent backend
# For video files
curl -X PUT http://${HOST_IP}:8000/api/v1/videos-for-search/<filename.mp4> \
-H "Content-Type: video/mp4" \
--data-binary @/path/to/video.mp4

# For RTSP stream (no credentials)
curl -s -X POST http://${HOST_IP}:8000/api/v1/rtsp-streams/add \
  -H "Content-Type: application/json" \
  -d '{"sensorUrl": "rtsp://<source_host>:<rtsp_port>/<stream_path>", "name": "<sensor_name>"}' | jq .

# For RTSP stream (with credentials + useful optional parameters)
# curl -s -X POST http://${HOST_IP}:8000/api/v1/rtsp-streams/add \
#   -H "Content-Type: application/json" \
#   -d '{
#     "sensorUrl": "rtsp://192.168.1.100:554/Streaming/Channels/101",
#     "name": "loading_dock_cam",
#     "username": "admin",
#     "password": "your_rtsp_password",
#     "location": "Warehouse loading dock",
#     "tags": "warehouse,dock,entrance"
#   }' | jq .
```

- Further verifications to determine if returned video sources match the user query. Each step to go deeper:
    - Check their source names, their video description / tags via the `vios` skill
    - Download screenshots using the `screenshot_url` of the best candidates (highest similarity scores) from the search hits (JSON results) to `/tmp`. Read them and verify if they correspond to the user query  

- Potentially retry by augmenting the user input with a lower similary threshold to include more results. This helps seeing if a clip of interest was filtered out due to a lower score

- Check if LLM/VLM are working:
```bash
# Ports are usually:
# - LLM: 30081
# - VLM: 30082
curl -s http://${HOST_IP}:${PORT}/v1/models | jq .

curl -s -X POST http://${HOST_IP}:${PORT}/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "<MODEL_NAME>", "max_tokens": 128, "messages": [{"role": "user", "content": "Hello!"}]}' | jq .
```

- Check if embeddings for that video source appear in Elasticsearch:
```bash
# List all indices with doc counts
curl -s "http://${HOST_IP}:9200/_cat/indices?h=index,docs.count,store.size&v"

# Count embeddings
curl -s "http://${HOST_IP}:9200/mdx-embed-filtered-2025-01-01/_count"

# Sample one embedding doc (without the vector)
curl -s "http://${HOST_IP}:9200/mdx-embed-filtered-2025-01-01/_search?size=1&pretty" \
  -H "Content-Type: application/json" \
  -d '{"_source": {"excludes": ["embedding"]}, "query": {"match_all": {}}}'
```
