# Remote WebRTC

Read when the browser and Pipecat agent are on different machines or networks. Localhost
testing does not prove remote media will work.

## Choose the Path

| Situation | Path |
| --- | --- |
| Same machine or trusted LAN | direct WebRTC with STUN |
| Remote GPU over SSH | tunnel signaling, then use reachable ICE candidates |
| NAT or restrictive firewall | coturn as TURN relay |
| UDP unavailable and TURN is not ready | WebSocket fallback |
| LiveKit Cloud | use LiveKit's documented connectivity path, not this Pipecat setup |

An SSH tunnel can carry HTTPS or WebSocket signaling. It does not make the remote GPU
host's UDP media reachable. WebRTC still needs a direct ICE path or TURN.

## Source of Truth

Read the current Pipecat
[Small WebRTC ICE configuration](https://docs.pipecat.ai/api-reference/server/services/transport/small-webrtc)
and [runner guide](https://docs.pipecat.ai/api-reference/server/utilities/runner/guide)
before generating code.

The development runner can add its default public STUN server, but it cannot inject a
custom TURN server from `bot.py`. For coturn, generate a signaling server that creates
`SmallWebRTCConnection` with the documented `ice_servers` configuration. Do not patch a
TURN URL into the default runner and assume it is used.

## Coturn

Run coturn on a host reachable from both browser and agent. Follow the current coturn and
network-provider documentation for installation and hardening. Configure:

- public DNS or IP advertised by the TURN server
- authenticated, expiring credentials where possible
- UDP and TCP TURN listeners
- TLS TURN on a firewall-friendly port for restrictive networks
- a relay port range opened in the host and cloud firewalls

Do not ship a static coturn password, config, or compose file in this skill. Keep TURN
credentials in `.env` placeholders and load them into the signaling layer. Keep public
TURN URLs in generated code or config with other endpoints.

Use `turn:` for UDP/TCP relay candidates and `turns:` for TLS according to the current
deployment. Prefer direct or UDP relay candidates. TCP/TLS relay is a compatibility
fallback and adds latency.

## Generated Project

When custom TURN is selected, `output-contract.md` must produce:

- `signaling_server.py` using the current documented `ice_servers` API
- matching browser and server ICE configuration
- TURN username and credential placeholders in `.env.example`
- exact HTTPS signaling/client startup instructions in the README
- a relay-only verification step that is disabled again after testing

`signaling_server.py` owns the HTTPS listener, browser client route, ICE configuration,
and WebRTC session creation. Keep pipeline construction in `bot.py` as an importable
factory when the current Pipecat docs support that shape. Do not start the default runner
and the custom signaling server as competing listeners.

For custom TURN, the README startup order is:

1. start or verify coturn
2. start `signaling_server.py`
3. open the HTTPS client URL served by that process

Do not start `bot.py` as a second process unless the current Pipecat architecture
explicitly requires it and the README documents how the processes communicate.

Generate a coturn Compose service only when this project owns the TURN deployment, and
derive it from current coturn deployment documentation. Otherwise document the external
TURN endpoint. Do not duplicate infrastructure the user already has.

DGX Spark and Jetson Thor are aarch64, and common coturn images publish `amd64` only.
Confirm an `arm64` image exists before generating that service on either platform. When
none does, use an external TURN endpoint and say why, rather than generating a service that
cannot start.

## Browser Security

Remote microphone access requires a secure browser context. Serve the client and signaling
endpoint over HTTPS with a valid certificate. `localhost` is the development exception.
Do not tell the user to bypass browser security warnings for production.

## WebSocket Fallback

Keep WebSocket available when intake selected both transports. It is a useful fallback
when corporate networks block UDP, but it is not a TURN replacement for WebRTC.

For an SSH-only development path, forward the agent's HTTP/WebSocket port and use the
WebSocket transport. Do not claim that an HTTP tunnel makes WebRTC media remote-safe.

## Verify

1. Direct LAN WebRTC succeeds.
2. A remote browser receives ICE candidates.
3. Test through a network where direct peer-to-peer traffic is unavailable.
4. Confirm the selected candidate type is `relay` during the TURN test.
5. Confirm bidirectional audio, not only signaling.
6. Return to normal ICE policy after the relay-only test.
7. Complete the spoken exchange in `operations/run.md`.

## Anti-Patterns

- Exposing only the signaling port and assuming media uses it.
- Putting coturn credentials in source code.
- Using the default Pipecat runner while claiming custom TURN is configured.
- Forcing relay-only mode for every production user.
- Treating WebSocket and TURN as the same transport mechanism.
