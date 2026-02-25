# Character-Based Leadership Elastic Constraint System (Dash)

Production-ready Dash + Plotly app that models a character-based elastic leadership system with a 2D to 3D transition for meta-leadership.

## Features

- 10 adjustable virtues (Judgment fixed at origin).
- 2D Character Mode:
  - radial virtue magnitudes in X-Y plane
  - elastic capacity circle
  - requested vs effective (compressed) contour
- 3D Meta-Leadership Mode:
  - sphere capacity boundary
  - Z axis: `+Up`, `-Down`, X-Y as leading across
- Global elastic capacity constraint:
  - 2D: `sqrt(sum(r_i^2)) <= MAX_CAPACITY`
  - 3D: `sqrt(sum(r_i^2 + z_i^2)) <= MAX_CAPACITY`
- Automatic nonlinear L2 proportional compression when exceeded.
- Optional Training Mode: capacity gradually increases when usage is balanced.
- Dark minimalist design for academic/professional presentation.

## Project Files

- `app.py` - Dash application and model logic
- `requirements.txt` - Python dependencies
- `Dockerfile` - container build
- `gunicorn_conf.py` - Gunicorn production settings

## Local Run (without Docker)

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Open: `http://localhost:8050`

## Docker Run (local)

```bash
docker build -t leadership-elastic:latest .
docker run --rm -p 8050:8050 --name leadership-elastic leadership-elastic:latest
```

Open: `http://localhost:8050`

## DigitalOcean Deployment (Droplet + Docker + Nginx)

Assume Ubuntu droplet and DNS A record already points to your server.

### 1) Provision server dependencies

```bash
sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg lsb-release nginx

# Docker Engine
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg

echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo $VERSION_CODENAME) stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
sudo usermod -aG docker $USER
newgrp docker
```

### 2) Upload app and build image

```bash
# From your workstation:
# scp -r leadership_elastic_app user@your_droplet_ip:/home/user/

cd /home/$USER/leadership_elastic_app
docker build -t leadership-elastic:latest .
```

### 3) Run container as service-style process

```bash
docker run -d \
  --name leadership-elastic \
  --restart unless-stopped \
  -p 127.0.0.1:8050:8050 \
  leadership-elastic:latest
```

Check logs:

```bash
docker logs -f leadership-elastic
```

### 4) Configure Nginx reverse proxy

Create `/etc/nginx/sites-available/leadership-elastic`:

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:8050;
        proxy_http_version 1.1;

        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";

        proxy_read_timeout 120s;
        proxy_connect_timeout 60s;
        proxy_send_timeout 120s;
    }
}
```

Enable and reload:

```bash
sudo ln -s /etc/nginx/sites-available/leadership-elastic /etc/nginx/sites-enabled/leadership-elastic
sudo nginx -t
sudo systemctl reload nginx
```

### 5) Optional HTTPS with Let's Encrypt

```bash
sudo apt-get install -y certbot python3-certbot-nginx
sudo certbot --nginx -d your-domain.com
```

Auto-renew check:

```bash
sudo systemctl status certbot.timer
```

## Container Update Workflow

```bash
cd /home/$USER/leadership_elastic_app
git pull  # if using git
docker build -t leadership-elastic:latest .
docker stop leadership-elastic && docker rm leadership-elastic
docker run -d --name leadership-elastic --restart unless-stopped -p 127.0.0.1:8050:8050 leadership-elastic:latest
```

## Notes

- App listens on port `8050`.
- Gunicorn serves Dash via `app:server`.
- Judgment is fixed at `(0,0,0)` and not user-adjustable.
