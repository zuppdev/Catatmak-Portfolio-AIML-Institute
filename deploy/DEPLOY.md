# Deployment Guide - Expense Tracker API

## Quick Start

### 1. Server Setup (Ubuntu/Debian)

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install dependencies
sudo apt install -y python3.11 python3.11-venv nginx certbot python3-certbot-nginx

# Create app directory
sudo mkdir -p /opt/expense-tracker
sudo chown $USER:$USER /opt/expense-tracker

# Clone/copy your code
cp -r . /opt/expense-tracker/
cd /opt/expense-tracker

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Generate API Key

```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

### 3. Configure Environment

```bash
cp .env.example .env
nano .env
```

Fill in:
```
EXPENSE_API_KEY=<your-generated-key>
ALLOWED_ORIGINS=https://your-frontend.com
```

### 4. Install Systemd Service

```bash
sudo cp deploy/expense-tracker.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable expense-tracker
sudo systemctl start expense-tracker

# Check status
sudo systemctl status expense-tracker
```

### 5. Configure Nginx + SSL

```bash
# Copy nginx config
sudo cp deploy/nginx.conf /etc/nginx/sites-available/expense-tracker

# Edit domain name
sudo nano /etc/nginx/sites-available/expense-tracker

# Enable site
sudo ln -s /etc/nginx/sites-available/expense-tracker /etc/nginx/sites-enabled/

# Get SSL certificate
sudo certbot --nginx -d your-domain.com

# Test and reload
sudo nginx -t
sudo systemctl reload nginx
```

### 6. Firewall

```bash
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw enable
```

## Using the API

### With API Key

```bash
curl -X POST https://your-domain.com/expense/text \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key-here" \
  -d '{"text": "makan bakso 20k"}'
```

### Response

```json
{
  "category": "makanan",
  "category_confidence": 1.0,
  "amount": 20000,
  "amount_formatted": "Rp 20.000",
  "currency": "IDR",
  "date": "2026-01-26"
}
```

## Security Checklist

- [x] API Key authentication
- [x] HTTPS only (via nginx)
- [x] CORS restricted to your domains
- [x] Rate limiting (nginx: 10 req/s)
- [x] Request size limits (10MB max)
- [x] Security headers
- [x] Systemd hardening (NoNewPrivileges, PrivateTmp)
- [ ] Regular security updates
- [ ] Log monitoring

## Monitoring

```bash
# View logs
sudo journalctl -u expense-tracker -f

# Check service
sudo systemctl status expense-tracker
```
