# Hosted analyser: single-tenant prototype

The desktop app remains `web_app.py` and should stay bound to localhost. The separate `hosted_api.py` entry point is an authenticated, stateless API: it does not expose the training queue, desktop photo history or shared current-photo state.

## Recommended free starting point

**Selected deployment: your Windows PC + Cloudflare Quick Tunnel.** Oracle is not being used. This keeps the current models on your computer and exposes only the separate authenticated API through a temporary HTTPS URL. No paid hosting account or Docker installation is needed for this prototype.

```powershell
python tools/public_api.py start --origin http://127.0.0.1:8767
python tools/public_api.py status
python tools/public_api.py stop
```

You can also use `START_PUBLIC_API.cmd` and `STOP_PUBLIC_API.cmd`. Start launches the API and tunnel in the background; closing the launcher window does not stop them. Use the stop command when finished. The first start downloads Cloudflare's official Windows executable and verifies the SHA256 digest published in its GitHub release. The executable is kept in ignored `.runtime/`; API credentials are generated in ignored `.env`; logs and process state stay in ignored `outputs/public-api/`. Credentials are never printed by the launcher. Use the local `.env` file to obtain your API key for a trusted client. Do not commit it or paste it into chat.

The allowed browser origin is currently `http://127.0.0.1:8767`. If your frontend moves, stop the managed service, then start again with the new `--origin` (repeat for multiple origins). Native and server-side clients do not need a CORS origin. The desktop UI's relative `/api/` calls are not automatically redirected: use the stateless hosted API contract below when integrating an external frontend.

The public URL changes after restarting the tunnel, and the service stops when your computer sleeps, shuts down or loses its Internet connection. Quick Tunnels are for testing/development, not guaranteed production uptime. Keep the models and Python dependencies installed locally. The public service does not expose your desktop UI, arbitrary files, training queue or model downloads. Only the two processes recorded by this launcher are stopped by its stop command.

References: [Cloudflare Quick Tunnels](https://developers.cloudflare.com/tunnel/get-started/quick-tunnels/), [official cloudflared releases](https://github.com/cloudflare/cloudflared/releases).

### Optional VM deployment, if hosting changes later

Use an **Oracle Cloud Always Free Ubuntu Ampere A1 VM, 1 OCPU, 6 GB RAM and a 50 GB boot volume**, in your account's home region, only if the console confirms the resources are within your remaining Always Free allowance. As checked on 13 November 2026, Oracle documents a total allowance equivalent to 2 OCPUs / 12 GB RAM and 200 GB block storage across an account. Existing resources count against it. Do not upgrade to Pay As You Go or select a paid fallback. Capacity is not guaranteed, and idle VMs may be reclaimed. This is a prototype option, not an uptime guarantee.

Create your own account at https://signup.cloud.oracle.com/. Oracle requires identity/card verification and agreement acceptance. Keep card details, passwords, private keys and API secrets out of chat and GitHub.

References: [Always Free limits](https://docs.oracle.com/en-us/iaas/Content/FreeTier/freetier_topic-Always_Free_Resources.htm), [signup requirements](https://www.oracle.com/cloud/free/faq/). Render's 512 MB free instance is too small for this model stack. New Docker Spaces on Hugging Face require a subscription; their CPU Basic label does not imply free signup access.

## What is prepared and what is not yet deployed

- Dockerfile: CPU-only PyTorch installation, non-root process, one Uvicorn worker.
- Compose: API on a private container network, Caddy HTTPS proxy on ports 80/443, 5 GiB API memory limit and one CPU.
- Mandatory random Bearer API key; explicit CORS origins; no wildcard browser access.
- A single active inference; concurrent requests get HTTP 429 with `Retry-After: 5`.
- 12 MiB image / 20 megapixel limits, request-size limit at the HTTPS proxy, in-memory analysis with no image retention.
- Model SHA256 verification before container startup; no model binaries or credentials in the container source or Git repository.

For the optional VM path, Docker is not installed in the current Windows workspace, so an actual container build, Arm runtime check, VM deployment, DNS and HTTPS certificate still need verification on that host. The selected PC + Quick Tunnel prototype does not need Docker or a VM; retrieve its current public URL with the status command above. Existing parking-model licences apply; the parking checkpoint is provided for evaluation/feedback and requests contact before commercial use/deployment.

## Server deployment steps

1. Create the Always Free VM, choose Ubuntu for Arm, and keep the SSH key locally. Record its public IP and SSH username. Do not use the 1 GB Micro VM for this application.
2. Configure OCI network and host firewall access for HTTP 80 and HTTPS 443; restrict SSH 22 to your administration address where possible. Do not expose the API container port 8080 directly.
3. Install [Docker Engine and the Compose plugin](https://docs.docker.com/engine/install/ubuntu/) from Docker's official Ubuntu instructions. Confirm the service is running.
4. Clone the code and enter its application folder:

```sh
git clone https://github.com/cdj27/PROJECT-OCS.git
cd PROJECT-OCS/Parking_Lite_YOLO26_Fixed
```

5. On the training computer, package the active weights with `python tools/package_hosted_models.py`. This creates the private `outputs/deployment/parking-models.tar.gz` archive. Transfer it to your VM using SSH/SCP; do not upload it to the public Git repository. In the application folder on the VM, extract it:

```sh
tar -xzf ~/parking-models.tar.gz
```

The archive contains the four required checkpoints, their manifest and provenance. All models are needed, including the locally trained occupancy head. Downloading only the three generic checkpoints does not reproduce the current analyser.

6. Point an API hostname to the VM's public IP, then copy `.env.example` to `.env`. Set:

```text
PARKING_API_KEY=<a new random secret, at least 32 characters>
PARKING_ALLOWED_ORIGINS=https://your-frontend-hostname
PARKING_MODELS_DIR=./models
API_DOMAIN=api.your-hostname
ACME_EMAIL=your-certificate-contact-email
```

Use a hostname you control. If you have no domain, choose a suitable free DNS hostname once the VM IP is known and verify its DNS and certificate issuance before relying on it. Origins contain only scheme and hostname/port, with no page path. Native/server clients do not need a CORS origin. The `.env` file is ignored by Git; protect it with `chmod 600 .env`. Generate the API key on the server, for example using Python's `secrets.token_urlsafe(48)`, and keep it private.

7. Start the services and check model verification and application logs:

```sh
docker compose up -d --build
docker compose ps
docker compose logs --tail=100 api https
```

Check `https://YOUR_API_HOST/readyz` returns `{"ready":true}`. `/healthz` only checks process liveness. The first analysis loads the models and can take longer. Test a real image through HTTPS before connecting the frontend. This CPU deployment is not a low-latency or multi-user service guarantee.

## API contract for external apps

**POST `/v1/analyze`**, `multipart/form-data`, header `Authorization: Bearer YOUR_API_KEY`.

| Field | Value |
|---|---|
| `file` | JPEG, PNG or WebP file |
| `quality` | `balanced` (default) or `accurate` |
| `engine` | `auto` (default), `white_lines` or `corner_brackets` |
| `include_image` | `false` (default) or `true` |

Response: `width`, `height`, `counts` (`total`, `occupied`, `available`), `spaces` (bay IDs, pixel-coordinate polygons, statuses, evidence and vehicle categories), `vehicles`, `diagnostic`. With `include_image=true`, the same JSON includes `annotated_image_base64` and `annotated_image_mime_type`. No separate upload call, shared photo ID or persistent result URL is needed.

```sh
curl https://YOUR_API_HOST/v1/analyze \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -F "file=@parking.jpg" \
  -F "quality=balanced" \
  -F "include_image=true"
```

Swagger documentation is available at `/docs`; use **Authorize** to provide the Bearer key. The desktop two-step `/api/upload` and `/api/analyze` routes remain on the local `web_app.py` service, not this hosted API.

The example client in `web/hosted-client.example.js` accepts the base URL and credential at runtime. Do not hard-code the shared API key into a public frontend bundle or GitHub Pages. For a public multi-user website, route calls through an authenticated server-side proxy or add individual user authentication first. This API's shared key is intended for trusted, single-tenant clients. Respect HTTP 429 and retry later rather than continuously resubmitting.

## Local API-only check

Set `PARKING_API_KEY` and run:

```sh
python -m uvicorn hosted_api:create_app --factory --host 127.0.0.1 --port 8080 --workers 1
```

Use the existing local model directory. This tests the hosted API contract without publishing the machine on the Internet.
