#!/bin/bash
cat > /tmp/app.conf << 'NGINXEOF'
server {
    listen 80;
    server_name _;

    location / {
        proxy_pass http://127.0.0.1:4000;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        client_max_body_size 15M;
    }
}
NGINXEOF

sudo cp /tmp/app.conf /etc/nginx/conf.d/app.conf
sudo nginx -t && sudo systemctl enable nginx && sudo systemctl start nginx && echo "nginx started OK"
